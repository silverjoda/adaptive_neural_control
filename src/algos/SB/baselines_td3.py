import gym
import sys
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TD3
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
import time
import random
import string
import socket
import numpy as np
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.noise import ActionNoise
from opensimplex import OpenSimplex

class SimplexNoise(ActionNoise):
    """
    A simplex action noise
    """
    def __init__(self, dim):
        super().__init__()
        self.idx = 0
        self.dim = dim
        self.noisefun = OpenSimplex(seed=int((time.time() % 1) * 10000000))

    def __call__(self) -> np.ndarray:
        self.idx += 1
        return [(self.noisefun.noise2d(x=self.idx / 10, y=i*10) + self.noisefun.noise2d(x=self.idx / 50, y=i*10)) * 0 for i in range(self.dim)]

    def __repr__(self) -> str:
        return 'Opensimplex Noise()'.format()

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

def lr_fun(step):
    return 1e-3 / np.power(1 + 9e-7, step)

if __name__ == "__main__":
    args = ["None", "perlin", "straight_rough"]
    if len(sys.argv) > 1:
        args = sys.argv

    from src.envs.bullet_nexabot.hexapod.hexapod import HexapodBulletEnv as env_fun

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 6000000,
              "batchsize": 60,
              "max_steps": 90,
              "gamma": 0.99,
              "policy_lr": 0.001,
              "weight_decay": 0.0001,
              "animate": True,
              "variable_velocity": False,
              "train": True,
              "terrain" : args[1],
              "r_type": args[2],
              "note": "Training: {}, {}, |Training with no penalty| ".format(args[1], args[2]),
              "ID": ID}

    print(params)
    TRAIN = False
    CONTINUE = False

    if TRAIN or socket.gethostname() == "goedel":
        if socket.gethostname() == "goedel":
            params["animate"] = False
        env = env_fun(animate=params["animate"],
                      max_steps=params["max_steps"],
                      step_counter=True,
                      terrain_name=params["terrain"],
                      training_mode=params["r_type"],
                      variable_velocity=False)

        n_actions = env.action_space.shape[-1]
        action_noise = SimplexNoise(n_actions)

        model = TD3('MlpPolicy',
                    env=env,
                    gamma=params["gamma"],
                    learning_rate=lr_fun,
                    buffer_size=1000000,
                    learning_starts=10000,
                    train_freq=1000,
                    gradient_steps=1000,
                    batch_size=128,
                    tau=0.005,
                    policy_delay=2,
                    action_noise=action_noise,
                    target_policy_noise=0.2,
                    target_noise_clip=0.5,
                    verbose=1,
                    tensorboard_log="./tb/{}/".format(ID),
                    policy_kwargs=dict(layers=[160, 140]))

        # Save a checkpoint every 1000000 steps
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='agents_cp/',
                                                 name_prefix=params["ID"], verbose=1)

        # Train the agent
        t1 = time.time()
        model.learn(total_timesteps=int(params["iters"]), callback=checkpoint_callback)
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
                  variable_velocity=False)

    if not TRAIN:
        #model = TD3.load("agents/ZFU_SB_policy.zip") # 4TD & 8CZ contactless:perlin:normal, U79 & BMT contactless:perlin:extreme, KIH turn_left, 266 turn_rigt
        model = TD3.load("agents_cp/JSK_5000000_steps.zip")  # 2Q5
    #print(evaluate_policy(model, env, n_eval_episodes=3))

    obs = env.reset()
    for _ in range(100):
        cum_rew = 0
        for i in range(800):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            env.render()
            if done:
                obs = env.reset()
                print(cum_rew)
                break
    env.close()