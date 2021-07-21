import gym
import pybulletgym
import time

env = gym.make('HumanoidPyBulletEnv-v0')
env.render() # call this before env.reset, if you want a window showing the environment
env.reset()  # should return a state vector if everything worked

while True:
    env.step([0]*17)
    time.sleep(0.004)