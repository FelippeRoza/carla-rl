import numpy as np

class hyperParameters(object):
    """Hyperparameters for RL agent"""

    def __init__(self):
        # ==============================================================================
        # -- Hyperparameters ----------------------------------------------------------
        # ==============================================================================
        self.state_size = [84, 84, 1]
        self.action_size = 6
        self.possible_actions = np.identity(6, dtype=int).tolist()
        self.learning_rate= 0.00025

        # Training parameters
        self.total_episodes = 5001  # INTIALLY  5000
        self.max_steps = 5000
        self.batch_size = 64

        # Fixed Q target hyper parameters
        self.max_tau = 5000  # tau is the C step where we update out target network -- INTIALLY 10000

        # exploration hyperparamters for ep. greedy. startegy
        self.explore_start = 1.0  # exploration probability at start
        self.explore_stop = 0.01  # minimum exploration probability
        self.decay_rate = 0.00005  # exponential decay rate for exploration prob

        # Q LEARNING hyperparameters
        self.gamma = 0.95  # Discounting rate
        self.pretrain_length = 100000  ## Number of experiences stored in the Memory when initialized for the first time --INTIALLY 100k
        self.memory_size = 100000  # Number of experiences the Memory can keep  --INTIALLY 100k
        self.load_memory = True
        self.memory_load_path = "rl/replay_memory/memory.pkl"
        self.memory_save_path = "rl/replay_memory/memory.pkl"

        # model saving
        self.model_save_frequency = 10 # frequency to save the model. 0 means not to save
        self.model_name = "DQNetwork"
        self.model_path = "rl/model"
        self.model_save_path = "rl/model/rl_model.ckpt"
