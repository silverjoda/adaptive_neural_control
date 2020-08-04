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
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
import time
import random
import string
import socket
import numpy as np

def make_env(params):
    def _init():
        env = env_fun(animate=params["animate"],
                      max_steps=params["max_steps"],
                      action_input=True,
                      latent_input=False)
        return env
    return _init

if __name__ == "__main__":
    args = ["None", "flat", "straight"]
    if len(sys.argv) > 1:
        args = sys.argv

    from src.envs.bullet_cartpole.hangpole_goal_cont_variable.hangpole_goal_cont_variable import HangPoleGoalContVariableBulletEnv as env_fun

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 1000000,
              "batchsize": 60,
              "max_steps": 100,
              "gamma": 0.99,
              "policy_lr": 0.001,
              "weight_decay": 0.0001,
              "ppo_update_iters": 1,
              "normalize_rewards": False,
              "animate": False,
              "variable_velocity": False,
              "train": True,
              "terrain" : args[1],
              "r_type": args[2],
              "note": "Training: {}, {}, |Training without contacts| ".format(args[1], args[2]),
              "ID": ID}

    print(params)
    TRAIN = True
    CONTINUE = False

    if TRAIN or socket.gethostname() == "goedel":
        n_envs = 6
        if socket.gethostname() == "goedel": n_envs = 8
        env = SubprocVecEnv([make_env(params) for _ in range(n_envs)], start_method='fork')
        policy_kwargs = dict(net_arch=[int(96), int(96)])

        model = A2C('MlpPolicy',
                    env=env,
                    learning_rate=params["policy_lr"],
                    verbose=1,
                    n_steps=30,
                    ent_coef=0.0,
                    vf_coef=0.5,
                    lr_schedule='linear',
                    tensorboard_log="./tb/{}/".format(ID),
                    full_tensorboard_log=False,
                    gamma=params["gamma"],
                    policy_kwargs=policy_kwargs)

        if CONTINUE:
            ID = "7X0"
            model = A2C.load("agents/{}_SB_policy.zip".format(ID))  # 4TD & 8CZ contactless:perlin:normal, U79 & BMT contactless:perlin:extreme, KIH turn_left, 266 turn_rigt
            model.tensorboard_log="./tb/{}/".format(ID),
            model.env = env

        # Save a checkpoint every 1000000 steps
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='agents_cp/',
                                                 name_prefix=params["ID"], verbose=1)

        eval_callback = EvalCallback(make_env(params)(), best_model_save_path='agents_best/{}/'.format(params["ID"]),
                                     eval_freq=100000,
                                     deterministic=True,
                                     render=False)

        callback = CallbackList([checkpoint_callback, eval_callback])

        # Train the agent
        t1 = time.time()
        model.learn(total_timesteps=int(params["iters"]), callback=callback)
        t2 = time.time()
        print("Training time: {}".format(t2-t1))
        print(params)
        model.save("agents/{}_SB_policy".format(params["ID"]))
        env.close()

    if socket.gethostname() == "goedel":
        exit()

    env = env_fun(animate=True,
                  max_steps=params["max_steps"],
                  action_input=True,
                  latent_input=False)

    if not TRAIN:
        model = A2C.load("agents/H2F_SB_policy.zip") # 4TD & 8CZ contactless:perlin:normal, U79 & BMT contactless:perlin:extreme, KIH turn_left, 266 turn_rigt

    obs = env.reset()
    for _ in range(100):
        cum_rew = 0
        for i in range(200):
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