# carla-rl

+ Reinforcement Learning implementations for Carla simulator [https://github.com/carla-simulator/carla](https://github.com/carla-simulator/carla).

+ So far, it is implemented the dueling deep-Q learning with prioritized experience replay

+ Tested on carla version 0.9.5

## How to run

+ Clone the project inside PythonAPI examples folder: `carla_folder/PythonAPI/examples`
+ Edit `rl_config.py` with the necessary hyperparameters 
+ Run `rl_agent.py`. The possible arguments are listed below
+ After training, the resulting model can be tested by running `rl_agent.py --test`

### rl_agent arguments

The arguments are adapted from `manual_control.py`, with minor changes

+ `'--test'`: test a trained model
+ `'-v', '--verbose'`: print debug information
+ `'--host'` (default='127.0.0.1'): IP of the host server
+ `'-p', '--port'` (default=2000): TCP port to listen to
+ `'-a', '--autopilot'`: enable autopilot
+ `'--res'` (default='800x600'): window resolution
+ `'--filter'` (default='vehicle.audi.tt'): 'actor filter
+ `'--rolename'` (default='hero'): actor role name
