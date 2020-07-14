import gym
import sys
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
import time
import random
import string
import socket

def make_env(params):
    def _init():
        env = env_fun(animate=params["animate"],
                      max_steps=params["max_steps"],
                      step_counter=True,
                      terrain_name=params["terrain"],
                      training_mode=params["r_type"],
                      variable_velocity=params["variable_velocity"])
        return env
    return _init

if __name__ == "__main__":
    args = ["None", "flat", "straight", "no_symmetry_pen"]
    if len(sys.argv) > 1:
        args = sys.argv

    from src.envs.bullet_nexabot.hexapod.hexapod_wip import HexapodBulletEnv as env_fun

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 9000000,
              "batchsize": 60,
              "max_steps": 100,
              "gamma": 0.99,
              "policy_lr": 0.003,
              "weight_decay": 0.0001,
              "ppo_update_iters": 1,
              "normalize_rewards": False,
              "symmetry_pen": args[3],
              "animate": False,
              "variable_velocity": False,
              "train": True,
              "terrain" : args[1],
              "r_type": args[2],
              "note": "Training: {}, {}, |Straight, just range difficulty increase| ".format(args[1], args[2]),
              "ID": ID}

    print(params)
    TRAIN = True

    if TRAIN or socket.gethostname() == "goedel":
        env = SubprocVecEnv([make_env(params) for _ in range(10)], start_method='fork')
        policy_kwargs = dict(net_arch=[int(96), int(96)])
        model = A2C('MlpPolicy', env, learning_rate=params["policy_lr"], verbose=1, n_steps=30, tensorboard_log="/tmp", gamma=params["gamma"], policy_kwargs=policy_kwargs)
        # Train the agent
        t1 = time.time()
        model.learn(total_timesteps=int(params["iters"]))
        t2 = time.time()
        print("Training time: {}".format(t2-t1))
        print(params)
        model.save("agents/{}_SB_policy".format(params["ID"]))
        env.close()

    if socket.gethostname() == "goedel":
        exit()

    env = env_fun(animate=True,
                  max_steps=params["max_steps"],
                  step_counter=True,
                  terrain_name=params["terrain"],
                  training_mode=params["r_type"],
                  variable_velocity=True)

    if not TRAIN:
        model = A2C.load("agents/T92_SB_policy.zip")  # 356, SFY, VZT

    print(evaluate_policy(model, env, n_eval_episodes=3))

    obs = env.reset()
    for _ in range(100):
        cum_rew = 0
        for i in range(800):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            env.render()
            if done:
                obs = env.reset()
                print(cum_rew)
                break
    env.close()