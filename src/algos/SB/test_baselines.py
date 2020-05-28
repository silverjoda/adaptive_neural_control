import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import time

from src.envs.bullet_cartpole.cartpole.cartpole import CartPoleBulletEnv as env
#from src.envs.bullet_cartpole.hangpole_goal.hangpole_goal import HangPoleGoalBulletEnv as env
#from src.envs.bullet_cartpole.double_cartpole_goal.double_cartpole_goal import DoubleCartPoleBulletEnv as env

TRAIN = False

if TRAIN:
    env_instance = env(animate=False, max_steps=200)
    env_instance_vec = DummyVecEnv([lambda: env_instance])

    model = PPO2('MlpPolicy', env_instance_vec, verbose=1, n_steps=300)
    model.learn(total_timesteps=200000)
    model.save("model")
    env_instance_vec.close()


env_instance = env(animate=True, max_steps=200)
env_instance_vec = DummyVecEnv([lambda: env_instance])
model = PPO2('MlpPolicy', env_instance_vec, verbose=1, n_steps=300)
model.load("model")
obs = env_instance_vec.reset()

for _ in range(100):
    cum_rew = 0
    for i in range(800):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env_instance_vec.step(action)
        cum_rew += reward
        env_instance_vec.render()
        time.sleep(0.01)
        if done:
            obs = env_instance_vec.reset()
            print(cum_rew)
            break

env_instance_vec.close()