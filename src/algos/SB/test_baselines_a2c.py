import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env
import time

#from src.envs.bullet_cartpole.cartpole.cartpole import CartPoleBulletEnv as env
#from src.envs.bullet_cartpole.hangpole_goal.hangpole_goal import HangPoleGoalBulletEnv as env
from src.envs.bullet_cartpole.double_cartpole_goal.double_cartpole_goal import DoubleCartPoleBulletEnv as env


TRAIN = True

if TRAIN:
    env_instance = env(animate=False, max_steps=300)
    model = A2C('MlpPolicy', env_instance, learning_rate=1e-3, verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(300000))
    model.save("a2c_mdl")
    del model
    env_instance.close()

env_instance = env(animate=True, max_steps=150)
# Load the trained agent
model = A2C.load("a2c_mdl")
print(evaluate_policy(model, env_instance, n_eval_episodes=3))

obs = env_instance.reset()
for _ in range(100):
    cum_rew = 0
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