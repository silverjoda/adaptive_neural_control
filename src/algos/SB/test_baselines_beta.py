import gym

from stable_baselines3 import PPO
import time

from src.envs.bullet_cartpole.cartpole.cartpole import CartPoleBulletEnv as env
#from src.envs.bullet_cartpole.hangpole_goal.hangpole_goal import HangPoleGoalBulletEnv as env
#from src.envs.bullet_cartpole.double_cartpole_goal.double_cartpole_goal import DoubleCartPoleBulletEnv as env
TRAIN = True

if TRAIN:
    env_instance = env(animate=False, max_steps=200)
    model = PPO('MlpPolicy', env_instance, verbose=1, n_steps=300)
    model.learn(total_timesteps=200000)
    model.save("model")
    env_instance.close()

env_instance = env(animate=True, max_steps=200)
model = PPO('MlpPolicy', env_instance, verbose=1, n_steps=300)
model.load("model")
obs = env_instance.reset()

for _ in range(100):
    cum_rew = 0.
    for i in range(800):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env_instance.step(action)
        cum_rew += reward
        env_instance.render()
        time.sleep(0.01)
        if done:
            obs = env_instance.reset()
            print(cum_rew)
            break

env_instance.close()