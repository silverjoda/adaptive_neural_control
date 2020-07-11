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
                      training_mode=params["r_type"])
        return env
    return _init

if __name__ == "__main__":
    args = ["None", "flat", "straight", "no_symmetry_pen"]
    if len(sys.argv) > 1:
        args = sys.argv

    from src.envs.bullet_nexabot.hexapod.hexapod_wip import HexapodBulletEnv as env_fun

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 500000,
              "batchsize": 60,
              "max_steps": 60,
              "gamma": 0.98,
              "policy_lr": 0.0007,
              "weight_decay": 0.0001,
              "ppo_update_iters": 1,
              "normalize_rewards": False,
              "symmetry_pen": args[3],
              "animate": False,
              "train": True,
              "terrain" : args[1],
              "r_type": args[2],
              "note": "Training: {}, {}".format(args[1], args[2]),
              "ID": ID}

    TRAIN = False

    if TRAIN or socket.gethostname() == "goedel":
        ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
        env = SubprocVecEnv([make_env(params) for _ in range(8)], start_method='fork')
        policy_kwargs = dict(net_arch=[int(96), int(96)])
        model = A2C('MlpPolicy', env, learning_rate=0.003, verbose=1, n_steps=30, tensorboard_log="/tmp", gamma=0.99, policy_kwargs=policy_kwargs)
        # Train the agent
        t1 = time.time()
        model.learn(total_timesteps=int(1200000))
        t2 = time.time()
        print("Training time: {}".format(t2-t1))
        model.save("agents/a2c_mdl")
        del model
        env.close()

    if socket.gethostname() == "goedel":
        exit()

    env = env_fun(animate=True,
                  max_steps=params["max_steps"],
                  step_counter=True,
                  terrain_name=params["terrain"],
                  training_mode=params["r_type"])
    # Load the trained agent
    model = A2C.load("agents/a2c_mdl")
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