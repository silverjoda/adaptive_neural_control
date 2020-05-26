import gym

from stable_baselines3 import PPO
import time

from src.envs.bullet_cartpole.double_cartpole_goal.double_cartpole_goal import DoubleCartPoleBulletEnv
TRAIN = False

if TRAIN:
    env = DoubleCartPoleBulletEnv(animate=False)
    model = PPO('MlpPolicy', env, verbose=1, n_steps=300)
    model.learn(total_timesteps=200000)
    model.save("model")
    env.close()

env = DoubleCartPoleBulletEnv(animate=True)
model = PPO('MlpPolicy', env, verbose=1, n_steps=300)
model.load("model")
obs = env.reset()

for _ in range(100):
    cum_rew = 0.
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