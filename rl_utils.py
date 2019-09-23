import tensorflow as tf
import numpy as np
import carla
import random
import time
import queue
import pickle
from PIL import Image

# ==============================================================================
# -- Classes -------------------------------------------------------------------
# ==============================================================================


# ==============================================================================
# -- Carla Related -------------------------------------------------------------
# ==============================================================================
class Sensors(object):
    """Class to keep track of all sensors added to the vehicle"""

    def __init__(self, world, vehicle):
        super(Sensors, self).__init__()
        self.world = world
        self.vehicle = vehicle
        self.camera_queue = queue.Queue() # queue to store images from buffer
        self.collision_flag = False # Flag for colision detection
        self.lane_crossed = False # Flag for lane crossing detection
        self.lane_crossed_type = '' # Which type of lane was crossed

        self.camera_rgb = self.add_sensors(world, vehicle, 'sensor.camera.rgb')
        self.collision = self.add_sensors(world, vehicle, 'sensor.other.collision')
        self.lane_invasion = self.add_sensors(world, vehicle, 'sensor.other.lane_invasion', sensor_tick = '0.5')

        self.sensor_list = [self.camera_rgb, self.collision, self.lane_invasion]

        self.collision.listen(lambda collisionEvent: self.track_collision(collisionEvent))
        self.camera_rgb.listen(lambda image: self.camera_queue.put(image))
        self.lane_invasion.listen(lambda event: self.on_invasion(event))

    def add_sensors(self, world, vehicle, type, sensor_tick = '0.0'):

        sensor_bp = self.world.get_blueprint_library().find(type)
        try:
            sensor_bp.set_attribute('sensor_tick', sensor_tick)
        except:
            pass
        if type == 'sensor.camera.rgb':
            sensor_bp.set_attribute('image_size_x', '100')
            sensor_bp.set_attribute('image_size_y', '100')

        sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=vehicle)
        return sensor

    def track_collision(self, collisionEvent):
        '''Whenever a collision occurs, the flag is set to True'''
        self.collision_flag = True

    def reset_sensors(self):
        '''Sets all sensor flags to False'''
        self.collision_flag = False
        self.lane_crossed = False
        self.lane_crossed_type = ''

    def on_invasion(self, event):
        '''Whenever the car crosses the lane, the flag is set to True'''
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.lane_crossed_type = text[0]
        self.lane_crossed = True

    def destroy_sensors(self):
        '''Destroy all sensors (Carla actors)'''
        for sensor in self.sensor_list:
            sensor.destroy()


# ==============================================================================
# -- Function approximation models ---------------------------------------------
# ==============================================================================

class DDDQNet():
    """This is a model for the Dueling Double Deep Q-learning method:
    The input is an image (state representation)
    It passes through 3 convnets, then it's flatened, then it's divided into 2 streams:
    One calculates the value function V(s) and other for the advantage function A(s,a)
    In the end an agregating layer outputs the Q values for each action, Q(s, a)"""

    # THIS IS AN UPDATED VERSION, WHERE THE NUMBER OF FILTERS IS CHANGED AND THE ACTIVATION FUNCTION AS WELL
    # Kernel initalizer is also changed to variance scaling and the optimizer to adamoptimizer

    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name
        self.possible_actions = np.identity(action_size, dtype=int).tolist()
        # we will use tf.variable_scope here to know which network we're usuing(DQN or target_net)
        # it will be useful when we update ourw-parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            # *state_size means that we take each element of state size in tuple hence is like if we wrote [None, 84,84,1]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name="IS_Weights")
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            # CNN OUTPUT FORMULA
            # W=(W−F+2P)/S  +1
            # where f is the receptive field (filter width),
            # p is the padding and s is the stride.

            """
            first Conv net CNN ELU
            input is 84*84*1
            gives us an output size of 20*20*32
            """
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32, kernel_size=[8, 8],
                                          strides=[4, 4],
                                          kernel_initializer=tf.variance_scaling_initializer(),
                                          padding="valid", activation=tf.nn.relu,
                                          name="conv1")

            """
            second conv net CNN ELU
            output size 9*9*64
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                          filters=64, kernel_size=[4, 4],
                                          strides=[2, 2],
                                          kernel_initializer=tf.variance_scaling_initializer(),
                                          padding="valid", activation=tf.nn.relu,
                                          name="conv2")

            """
            Third conv net CNN ELU
            out put size 7*7*64
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2,
                                          filters=64, kernel_size=[3, 3],
                                          strides=[1, 1],
                                          kernel_initializer=tf.variance_scaling_initializer(),
                                          padding="valid", activation=tf.nn.relu,
                                          name="conv3")

            """Flatten layer"""
            self.flatten = tf.layers.flatten(self.conv3)

            """Value function V(s) computation layers"""
            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=1024, activation=tf.nn.relu,  # 7*7*64 = 3136
                                            kernel_initializer=tf.variance_scaling_initializer(),
                                            name="value_fc")
            self.value = tf.layers.dense(inputs=self.value_fc, units=1,
                                         activation=None,
                                         kernel_initializer=tf.variance_scaling_initializer(),
                                         name="value")

            """Advantage function A(s, a) computation layers"""
            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=1024,  # was previously 256
                                                activation=tf.nn.relu,
                                                kernel_initializer=tf.variance_scaling_initializer(),
                                                name="advantage_fc")
            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.variance_scaling_initializer(),
                                             name="advantage")

            """
            Agregating layer
            Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            """
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
            self.output = tf.identity(self.output, name="output")
            # Q is out predicted value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            # the loss is modified because of PER

            """We want to take in priority experience where there is a big difference between
            our prediction and the TD target, since it means that we have a lot to learn about it."""
            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating sumtree
            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
            self.loss_2 = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_Q, predictions=self.Q))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.optimizer_2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_2)


    def predict_action(self, sess, explore_start, explore_stop, decay_rate, decay_step, state, action_size):
        # Epsilon greedy strategy: given state s, choose action a ep. greedy
        exp_tradeoff = np.random.rand()
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if (explore_probability > exp_tradeoff):
            action_int = np.random.choice(action_size)
            action = self.possible_actions[action_int]
        else:
            # get action from Q-network: neural network estimates the Q values
            Qs = sess.run(self.output, feed_dict={self.inputs_: state.reshape((1, *state.shape))})

            # choose the best Q value from the discrete action space (argmax)
            action_int = np.argmax(Qs)
            action = self.possible_actions[int(action_int)]

        return action_int, action, explore_probability


# ==============================================================================
# -- Prioritized Replay --------------------------------------------------------
# ==============================================================================

class SumTree(object):
    """
        This SumTree code is modified version of Morvan Zhou:
        https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py"""

    data_pointer = 0

    # initialise tree with all nodes = 0 and data with all values =0

    def __init__(self, capacity):

        self.capacity = capacity
        # number of leaf nodes that contains experiences
        # generate the tree with all nodes values =0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity

        self.tree = np.zeros(
            2 * capacity - 1)  # was initally np.zeroes, but after making memory_size>pretain_length, it had to be adjusted

        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):

        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            # overwrite
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # change = new priority - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through the tree // update whole tree
        while tree_index != 0:
            """
                        Here we want to access the line above
                        THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                            0
                           / \
                          1   2
                         / \ / \
                        3  4 5  [6]

                        If we are in leaf at index 6, we updated the priority score
                        We need then to update index 2 node
                        So tree_index = (tree_index - 1) // 2
                        tree_index = (6-1)//2
                        tree_index = 2 (because // round the result)
                        """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        # here we get the leaf_index, priority value of that leaf and experience associated with that index
        """
                        Tree structure and array storage:
                        Tree index:
                             0         -> storing priority sum
                            / \
                          1     2
                         / \   / \
                        3   4 5   6    -> storing priority for experiences
                        Array type for storing:
                        [0,1,2,3,4,5,6]
                        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        # data_index is the child index that we want ro get the data from, leaf index is it's parent ndex
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class Memory(object):
    """
    This SumTree code is modified version of:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """

    def __init__(self, capacity, pretrain_length, action_size):
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """

        self.tree = SumTree(capacity)
        self.pretrain_length = pretrain_length
        self.action_size = action_size
        self.possible_actions = np.identity(action_size, dtype=int).tolist()
        # hyperparamters
        self.absolute_error_upper = 1.  # clipped abs error
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
        self.PER_b_increment_per_sampling = 0.001

    def store(self, experience):
        """
         Store a new experience in the tree with max_priority
         When training the priority is to be ajusted according with the prediction error
        """
        # find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        # use minimum priority =1
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):
        # create sample array to contain minibatch
        memory_b = []
        if n > self.tree.capacity:
            print("Sample number more than capacity")
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        # calc the priority segment, divide Range into n ranges
        priority_segment = self.tree.total_priority / n

        # increase PER_b each time we sample a minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        # calc max_Weights
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        # print(np.min(self.tree.tree[-self.tree.capacity:]))
        # print(self.tree.total_priority)
        # print("pmin =" , p_min)
        max_weight = (p_min * n) ** (-self.PER_b)
        # print("max weight =" ,max_weight)

        for i in range(n):
            # A value is uniformly sampled from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)
            # print("priority =", priority)

            sampling_probabilities = priority / self.tree.total_priority
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            # print("weights =", b_ISWeights[i,0])
            # print(b_ISWeights.shape) shape(64,1)

            b_idx[i] = index
            experience = [data]
            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):

        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def fill_memory(self, map, vehicle, camera_queue, sensors):
        print("Started to fill memory")
        reset_environment(map, vehicle, sensors)

        for i in range(self.pretrain_length):
            # random action
            if i % 500 == 0:
                print(i, "experiences stored")
            state = process_image(camera_queue)
            action_int = np.random.choice(self.action_size)
            action = self.possible_actions[action_int]
            car_controls = map_action(action_int)
            vehicle.apply_control(car_controls)
            time.sleep(0.25)

            reward = compute_reward(vehicle, sensors)
            done = isDone(reward)
            next_state = process_image(camera_queue)
            experience = state, action, reward, next_state, done
            self.store(experience)

            if done:
                reset_environment(map, vehicle, sensors)
            else:
                state = next_state

    def save_memory(self, filename, object):
        handle = open(filename, "wb")
        pickle.dump(object, handle)

    def load_memory(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


# ==============================================================================
# -- Functions -----------------------------------------------------------------
# ==============================================================================

def get_split_batch(batch):
    '''memory.sample() returns a batch of experiences, but we want an array
    for each element in the memory (s, a, r, s', done)'''

    states_mb = np.array([each[0][0] for each in batch], ndmin=3)
    # print(states_mb.shape) #shape 64*84*84*1 after reshaping im_final -- 64 is the batch size
    actions_mb = np.array([each[0][1] for each in batch])
    # print(actions_mb.shape)
    rewards_mb = np.array([each[0][2] for each in batch])
    # print(rewards_mb.shape) #shape (64,)
    next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
    # print(next_states_mb.shape)
    dones_mb = np.array([each[0][4] for each in batch])

    return states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb

def map_action(action):
    """ maps discrete actions into actual values to control the car"""
    control = carla.VehicleControl()
    control.throttle = 0.3
    control.brake = 0.0
    if action == 0:
        control.throttle = 0.5
        control.brake = 0.0
    if action == 1:
        control.steer = 0.0
    if action == 2:
        control.throttle = 0.5
        control.steer = 0.5
    if action == 3:
        control.throttle = 0.5
        control.steer = -0.5
    if action == 4:
        control.throttle = 0.5
        control.steer = 0.25
    if action == 5:
        control.throttle = 0.5
        control.steer = -0.25
    return control


def reset_environment(map, vehicle, sensors):
    ''' Set the vehicle velocities to 0 and move it to a spawn point'''
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    time.sleep(1) # wait for the car to stop
    spawn_points = map.get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
    vehicle.set_transform(spawn_point)
    time.sleep(2) # wait for the car to spawn
    sensors.reset_sensors() # set sensor flags to False


def process_image(queue):
    '''get the image from the buffer and process it. It's the state for vision-based systems'''
    image = queue.get()
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image = Image.fromarray(array).convert('L') # grayscale conversion
    image = np.array(image.resize((84, 84))) # convert to numpy array
    image = np.reshape(image, (84, 84, 1)) # reshape image
    return image


def compute_reward(vehicle, sensors):#, collision_sensor, lane_sensor):
    max_speed = 14
    min_speed = 2
    speed = vehicle.get_velocity()
    vehicle_speed = np.linalg.norm([speed.x, speed.y, speed.z])

    speed_reward = (abs(vehicle_speed) - min_speed) / (max_speed - min_speed)
    lane_reward = 0

    if (vehicle_speed > max_speed) or (vehicle_speed < min_speed):
        speed_reward = -0.05

    if sensors.lane_crossed:
        if sensors.lane_crossed_type == "'Broken'" or sensors.lane_crossed_type == "'NONE'":
            lane_reward = -0.5
            sensors.lane_crossed = False

    if sensors.collision_flag:
        return -1

    else:
        return speed_reward + lane_reward


def isDone(reward):
    '''Return True if the episode is finished'''
    if reward <= -1:
        return True
    else:
        return False
