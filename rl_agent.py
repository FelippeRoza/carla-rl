import sys
sys.path.append("..") # Add parent directory to path
import manual_control
import carla
import time
import random
import argparse
import logging
import pygame
from multiprocessing import Process
import sqlalchemy as db
import numpy as np
import tensorflow as tf
import queue # to get sensor data
import rl_utils
from rl_utils import DDDQNet, SumTree, Memory, map_action, reset_environment
from rl_utils import process_image, compute_reward, isDone, get_split_batch
from rl_config import hyperParameters

def render(clock, world, display):
    clock.tick_busy_loop(30) # this sets the maximum client fps
    world.tick(clock)
    world.render(display)
    pygame.display.flip()


def update_target_graph():
    # This function helps to copy one set of variables to another
    # In our case we use it when we want to copy the parameters of DQN to Target_network
    # Thanks to Arthur Juliani https://github.com/awjuliani

    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")
    op_holder = []

    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder


def init_tensorflow():
    # ==============================================================================
    # -- tensorflow init
    configProto = tf.ConfigProto()
    configProto.gpu_options.allow_growth = True
    # reset tensorflow graph
    tf.reset_default_graph()
    return configProto


def train_loop(rl_config, vehicle, map, sensors):

    configProto = init_tensorflow()
    # instantiate the DQN target networks
    DQNetwork = DDDQNet(rl_config.state_size, rl_config.action_size, rl_config.learning_rate, name=rl_config.model_name)
    TargetNetwork = DDDQNet(rl_config.state_size, rl_config.action_size, rl_config.learning_rate, name="TargetNetwork")

    #tensorflow summary for tensorboard visualization
    writer = tf.summary.FileWriter(".rl/summary")
    # losses
    tf.summary.scalar("Loss", DQNetwork.loss)
    tf.summary.scalar("Hubor_loss", DQNetwork.loss_2)
    tf.summary.histogram("ISWeights", DQNetwork.ISWeights_)
    write_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    # initialize memory and fill it with examples, for prioritized replay
    memory = Memory(rl_config.memory_size, rl_config.pretrain_length, rl_config.action_size)
    if rl_config.load_memory:
        memory = memory.load_memory(rl_config.memory_load_path)
        print("Memory Loaded")
    else:
        #this can take a looong time
        memory.fill_memory(map, vehicle, sensors.camera_queue, sensors)
        memory.save_memory(rl_config.memory_save_path, memory)

    # Reinforcement Learning loop
    with tf.Session(config=configProto) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        m = 0
        decay_step = 0
        tau = 0
        # update param, of target network rith DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)

        for episode in range(1, rl_config.total_episodes):
            # move the vehicle to a spawn_point and return state
            reset_environment(map, vehicle, sensors)
            state = process_image(sensors.camera_queue)
            done = False
            start = time.time()
            episode_reward = 0
            # save the model from time to time
            if rl_config.model_save_frequency:
                if episode % rl_config.model_save_frequency == 0:
                    save_path = saver.save(sess, rl_config.model_save_path)

            for step in range(rl_config.max_steps):
                tau += 1
                decay_step += 1

                action_int, action, explore_probability = DQNetwork.predict_action(sess, rl_config.explore_start,
                                                                         rl_config.explore_stop, rl_config.decay_rate,
                                                                         decay_step, state, rl_config.action_size)
                car_controls = map_action(action_int)
                vehicle.apply_control(car_controls)
                time.sleep(0.25)
                next_state = process_image(sensors.camera_queue)
                reward = compute_reward(vehicle, sensors)
                episode_reward += reward
                done = isDone(reward)
                experience = state, action, reward, next_state, done
                memory.store(experience)

                # Lets learn
                # First we need a mini-batch with experiences (s, a, r, s', done)
                tree_idx, batch, ISWeights_mb = memory.sample(rl_config.batch_size)
                s_mb, a_mb, r_mb, next_s_mb, dones_mb = get_split_batch(batch)

                # Get Q values for next_state from the DQN and TargetNetwork
                q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_s_mb})
                q_target_next_state = sess.run(TargetNetwork.output,
                                               feed_dict={TargetNetwork.inputs_: next_s_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise Q_target = r + gamma * Qtarget(s',a')
                target_Qs_batch = []
                for i in range(0, len(dones_mb)):
                    terminal = dones_mb[i]
                    # we got a'
                    action = np.argmax(q_next_state[i])
                    # if we are in a terminal state. only equals reward
                    if terminal:
                        target_Qs_batch.append((r_mb[i]))
                    else:
                        # take the Q taregt for action a'
                        target = r_mb[i] + rl_config.gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                _, _, loss, loss_2, absolute_errors = sess.run(
                    [DQNetwork.optimizer, DQNetwork.optimizer_2, DQNetwork.loss, DQNetwork.loss_2,
                     DQNetwork.absolute_errors],
                    feed_dict={DQNetwork.inputs_: s_mb,
                               DQNetwork.target_Q: targets_mb,
                               DQNetwork.actions_: a_mb,
                               DQNetwork.ISWeights_: ISWeights_mb})

                # update replay memory priorities
                memory.batch_update(tree_idx, absolute_errors)

                if tau > rl_config.max_tau:
                    update_target = update_target_graph()
                    sess.run(update_target)
                    m += 1
                    tau = 0
                    print("model updated")

                state = next_state

                if done:
                    print(episode, 'episode finished. Episode total reward:', episode_reward)
                    break

def test_loop(rl_config, vehicle, map, sensors):
    configProto = init_tensorflow()
    with tf.Session(config=configProto) as sess:

        saver = tf.train.import_meta_graph(rl_config.model_save_path + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(rl_config.model_path))

        if saver is None:
            print("did not load")

        graph = tf.get_default_graph()
        inputs_ = graph.get_tensor_by_name(rl_config.model_name + "/inputs:0")
        output = graph.get_tensor_by_name(rl_config.model_name + "/output:0")

        episode_reward = 0
        reset_environment(map, vehicle, sensors)

        while True:
            state = process_image(sensors.camera_queue)
            Qs = sess.run(output, feed_dict={inputs_: state.reshape((1, *state.shape))})
            action_int = np.argmax(Qs)
            #print(Qs)
            #print(action_int)

            car_controls = map_action(action_int)
            vehicle.apply_control(car_controls)
            reward = compute_reward(vehicle, sensors)
            episode_reward += reward
            done = isDone(reward)

            if done:
                print("EPISODE ended", "TOTAL REWARD {:.4f}".format(episode_reward))
                reset_environment(map, vehicle, sensors)
                total_reward = 0

            else:
                time.sleep(0.25)

def control_loop(vehicle_id, host, port, test_flag):
    actor_list = []
    try:
        #setup Carla
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        world = client.get_world()
        map = world.get_map()
        vehicle = next((x for x in world.get_actors() if x.id == vehicle_id), None) #get the vehicle actor according to its id
        sensors = rl_utils.Sensors(world, vehicle)
        rl_config = hyperParameters() # algorithm hyperparameters

        if test_flag:
            test_loop(rl_config, vehicle, map, sensors)
        else:
            train_loop(rl_config, vehicle, map, sensors)

    finally:
        for actor in actor_list:
            actor.destroy()


def render_loop(args):
    #loop responsible for rendering the simulation client
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = manual_control.HUD(args.width, args.height)
        world = manual_control.World(client.get_world(), hud, args.filter, args.rolename)

        p = Process(target=control_loop, args=(world.player.id, args.host, args.port, args.test, ))
        p.start()
        # p.join()
        clock = pygame.time.Clock()

        while True:
            render(clock, world, display) #pygame output update
    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA RL')
    argparser.add_argument(
        '--test',
        action='store_true',
        dest='test',
        help='test a trained model')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x600',
        help='window resolution (default: 800x600)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.tt',
        help='actor filter (default: "vehicle.audi.tt")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        render_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
