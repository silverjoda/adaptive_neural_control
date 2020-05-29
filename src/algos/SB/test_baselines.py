import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN
from stable_baselines.common.evaluation import evaluate_policy
import time

from src.envs.bullet_cartpole.cartpole.cartpole import CartPoleBulletEnv as env
#from src.envs.bullet_cartpole.hangpole_goal.hangpole_goal import HangPoleGoalBulletEnv as env
#from src.envs.bullet_cartpole.double_cartpole_goal.double_cartpole_goal import DoubleCartPoleBulletEnv as env

TRAIN = False

if TRAIN:
    env_instance = env(animate=False, max_steps=300)
    #env_instance_vec = DummyVecEnv([lambda: env_instance])

    model = DQN('MlpPolicy', env_instance, learning_rate=1e-3, prioritized_replay=True, verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(50000))
    model.save("dqn_mdl")
    del model
    env_instance.close()

env_instance = env(animate=True, max_steps=200)
# Load the trained agent
model = DQN.load("dqn_mdl")
#print(evaluate_policy(model, env_instance, n_eval_episodes=10))

obs = env_instance.reset()
for _ in range(100):
    cum_rew = 0
    for i in range(800):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env_instance.step(action)
        cum_rew += reward
        env_instance.render()
        time.sleep(0.02)
        if done:
            obs = env_instance.reset()
            print(cum_rew)
            break

env_instance.close()