import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import time

from src.envs.bullet_cartpole.cartpole.cartpole import CartPoleBulletEnv as env
#from src.envs.bullet_cartpole.hangpole_goal.hangpole_goal import HangPoleGoalBulletEnv as env
#from src.envs.bullet_cartpole.double_cartpole_goal.double_cartpole_goal import DoubleCartPoleBulletEnv as env

TRAIN = True

if TRAIN:
    env = env(animate=False, max_steps=200)
    env = DummyVecEnv([lambda: env])

    model = PPO2('MlpPolicy', env, verbose=1, n_steps=300)
    model.learn(total_timesteps=200000)
    model.save("model")
    env.close()

env = env(animate=False, max_steps=200)
env = DummyVecEnv([lambda: env])
model = PPO2('MlpPolicy', env, verbose=1, n_steps=300)
model.load("model")
obs = env.reset()

for _ in range(100):
    cum_rew = 0
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cum_rew += reward
        env.render()
        time.sleep(0.01)
        if done:
            obs = env.reset()
            print(cum_rew)
            break

env.close()