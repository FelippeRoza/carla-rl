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
from rl_utils import DDDQNet, SumTree, Memory, map_action, reset_environment, process_image, compute_reward, isDone
from rl_config import hyperParameters

def render(clock, world, display):
    clock.tick_busy_loop(30) # this sets the maximum client fps
    world.tick(clock)
    world.render(display)
    pygame.display.flip()


def update_target_graph():
    # This function helps to copy one set of variables to another
    # In our case we use it when we want to copy the parameters of DQN to Target_network
    # Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani

    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")
    op_holder = []

    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder


def control_loop(vehicle_id, host, port):
    actor_list = []
    try:
        #setup Carla
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        world = client.get_world()
        map = world.get_map()
        vehicle = next((x for x in world.get_actors() if x.id == vehicle_id), None) #get the vehicle actor according to its id

        sensors = rl_utils.Sensors(world, vehicle)

        rl_config = hyperParameters(load_memory = False) # algorithm hyperparameters

        # ==============================================================================
        # -- tensorflow init
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        # reset tensorflow graph
        tf.reset_default_graph()
        # instantiate the DQNetwork
        DQNetwork = DDDQNet(rl_config.state_size, rl_config.action_size, rl_config.learning_rate, name="DQNetwork")
        # instantiate the target network
        TargetNetwork = DDDQNet(rl_config.state_size, rl_config.action_size, rl_config.learning_rate, name="TargetNetwork")

        #tensorflow summary for tensorboar visualization
        writer = tf.summary.FileWriter("./summaries/waypoint/5th")
        # losses
        tf.summary.scalar("Loss", DQNetwork.loss)
        tf.summary.scalar("Hubor_loss", DQNetwork.loss_2)
        tf.summary.histogram("ISWeights", DQNetwork.ISWeights_)
        write_op = tf.summary.merge_all()
        saver = tf.train.Saver()

        # initialize memory and fill it with examples, for prioritized replay
        memory = Memory(rl_config.memory_size, rl_config.pretrain_length, rl_config.action_size)
        if rl_config.load_memory:
            memory = memory.load_memory("./replay_memory/memory_aftertrain.pkl")
            print("Memory Loaded")
        else:
            #this can take a looong time
            memory.fill_memory(map, vehicle, sensors.camera_queue, sensors)
            memory.save_memory("./replay_memory/memory.pkl", memory)

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

            for episode in range(rl_config.total_episodes):
                # move the vehicle to a spawn_point and return state
                reset_environment(map, vehicle, sensors)
                state = process_image(sensors.camera_queue)
                done = False
                start = time.time()
                score = 0

                for step in range(rl_config.max_steps):
                    tau += 1
                    decay_step += 1

                    action_int, action, explore_probability = DQNetwork.predict_action(sess, rl_config.explore_start,
                                                                             rl_config.explore_stop, rl_config.decay_rate,
                                                                             decay_step, state, rl_config.action_size)
                    car_controls = map_action(action_int)
                    vehicle.apply_control(car_controls)
                    time.sleep(0.25)

                    reward = compute_reward()
                    done = isDone(reward)

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

        p = Process(target=control_loop, args=(world.player.id, args.host, args.port, ))
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
