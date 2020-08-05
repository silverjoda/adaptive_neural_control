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
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
import time
import random
import string
import socket
import numpy as np

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

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
    args = ["None", "flat", "straight"]
    if len(sys.argv) > 1:
        args = sys.argv

    from src.envs.bullet_nexabot.hexapod.hexapod import HexapodBulletEnv as env_fun

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 40000000,
              "batchsize": 60,
              "max_steps": 90,
              "gamma": 0.97,
              "policy_lr": 0.002,
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
    TRAIN = False
    CONTINUE = False

    if TRAIN or socket.gethostname() == "goedel":
        n_envs = 1
        if socket.gethostname() == "goedel": n_envs = 8
        env = SubprocVecEnv([make_env(params) for _ in range(n_envs)], start_method='fork')
        policy_kwargs = dict(net_arch=[int(96), int(96)])

        if CONTINUE:
            ID = "FXX" # FXX
            print("Continuing training with : {}".format(ID))
            params["ID"] = ID
            model = A2C.load("agents/{}_SB_policy.zip".format(ID))  # 4TD & 8CZ contactless:perlin:normal, U79 & BMT contactless:perlin:extreme, KIH turn_left, 266 turn_rigt
            model.env = env
            model.tensorboard_log="./tb/{}/".format(ID)
            model.lr_schedule = 'linear'
            model.learning_rate = params["policy_lr"] / 5.
            model.n_steps = 30
            model.ent_coef = 0.0
            model.vf_coef = 0.5
        else:
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

        # Save a checkpoint every 1000000 steps
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='agents_cp/',
                                                 name_prefix=params["ID"], verbose=1)

        custom_callback = CustomCallback()
        callback = CallbackList([checkpoint_callback])

        # TODO: Custom callback for symmetry <-

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
                  step_counter=True,
                  terrain_name=params["terrain"],
                  training_mode=params["r_type"],
                  variable_velocity=False)

    if not TRAIN:
        model = A2C.load("agents/FXX_SB_policy.zip") # 4TD & 8CZ contactless:perlin:normal, U79 & BMT contactless:perlin:extreme, KIH turn_left, 266 turn_rigt
        #model = A2C.load("agents_cp/FXX_2400000_steps.zip")  # 2Q5
    #print(evaluate_policy(model, env, n_eval_episodes=3))

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