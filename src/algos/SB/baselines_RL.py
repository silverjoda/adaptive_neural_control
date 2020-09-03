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
import argparse
import yaml

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
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        for _ in range(self.model.n_steps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            obs, rewards, dones, infos = self.env.step(clipped_actions)

            self.model.num_timesteps += self.n_envs

            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions, dtype=self.env.action_space.dtype).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        true_rewards = np.copy(mb_rewards)
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # discount/bootstrap off value fn

        # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])
        true_rewards = true_rewards.reshape(-1, *true_rewards.shape[2:])

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

def make_env(params, env_fun):
    def _init():
        env = env_fun(params)
        return env
    return _init

def parse_args():
    parser = argparse.ArgumentParser(description='Pass in parameters. ')
    # parser.add_argument('--n_steps', type=int, required=False, help='Number of training steps .')
    # parser.add_argument('--terrain_type', type=str, default="flat", required=False, help='Type of terrain for training .')
    # parser.add_argument('--lr', type=int, default=0.001, required=False, help='Learning rate .')
    # parser.add_argument('--batchsize', type=int, default=32, required=False, help='Batchsize .')
    parser.add_argument('--algo_config', type=str, default="default_algo_config.yaml", required=False,
                        help='Algorithm config flie name .')
    parser.add_argument('--env_config', type=str, default="default_env_config.yaml", required=False,
                        help='Env config flie name .')

    args = parser.parse_args()
    return args.__dict__

def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def import_env(name):
    env_fun = None
    if name == "hexapod":
        from src.envs.bullet_nexabot.hexapod.hexapod import HexapodBulletEnv as env_fun
    if name == "quadrotor":
        from src.envs.bullet_quadrotor.quadrotor import QuadrotorBulletEnv as env_fun
    if name == "buggy":
        from src.envs.bullet_buggy.buggy import BuggyBulletEnv as env_fun
    assert env_fun is not None, "Env name not found, exiting. "
    return env_fun

def make_model(config):
    A2C('MlpPolicy',
        env=env,
        learning_rate=algo_config["policy_lr"],
        verbose=algo_config["verbose"],
        n_steps=algo_config["n_steps"],
        ent_coef=algo_config["ent_coef"],
        vf_coef=algo_config["vf_coef"],
        lr_schedule=algo_config["lr_schedule"],
        tensorboard_log="./tb/{}/".format(ID),
        full_tensorboard_log=algo_config["full_tensorboard_log"],
        gamma=algo_config["gamma"],
        policy_kwargs=algo_config["policy_kwargs"])
    return None

if __name__ == "__main__":
    # Read script input arguments
    args = parse_args()
    print(args)

    # Read configurations from yaml
    algo_config = read_config(args["algo_config"])
    print(algo_config)
    env_config = read_config("env_config")
    print(env_config)

    # Import correct env by name
    env_fun = import_env(env_config["env_name"])

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

    if algo_config["is_train"] or socket.gethostname() == "goedel":
        n_envs = 1
        if socket.gethostname() == "goedel": n_envs = 10
        env = SubprocVecEnv([make_env(env_config, env_fun) for _ in range(n_envs)], start_method='fork')

        model = make_model(algo_config)

        # Save a checkpoint every 1000000 steps
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='agents_cp/',
                                                 name_prefix=ID, verbose=1)

        custom_callback = CustomCallback()
        callback = CallbackList([checkpoint_callback])

        # Train the agent
        t1 = time.time()
        model.learn(total_timesteps=int(algo_config["iters"]), callback=callback)
        t2 = time.time()
        print("Training time: {}".format(t2-t1))
        print(algo_config)
        model.save("agents/{}_SB_policy".format(ID))
        env.close()

    if socket.gethostname() == "goedel":
        exit()

    env = make_env(env_config, env_fun)

    if not algo_config["is_train"]:
        model = A2C.load("agents/{}".format(algo_config["test_agent"]))

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