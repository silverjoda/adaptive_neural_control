from stable_baselines import PPO2, A2C, TD3, SAC
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
import src.my_utils as my_utils
import time
import random
import string
import socket
import argparse
import yaml
import os
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Pass in parameters. ')
    parser.add_argument('--train',  action='store_true', required=False,
                        help='Flag indicating whether the training process is to be run. ')
    parser.add_argument('--test', action='store_true', required=False,
                        help='Flag indicating whether the testing process is to be run. ')
    parser.add_argument('--animate', action='store_true', required=False,
                        help='Flag indicating whether the environment will be rendered. ')
    parser.add_argument('--test_agent_path', type=str, default=".", required=False,
                        help='Path of test agent. ')
    parser.add_argument('--algo_config', type=str, default="configs/td3_default_config.yaml", required=False,
                        help='Algorithm config file name. ')
    parser.add_argument('--env_config', type=str, required=True,
                        help='Env config file name. ')
    parser.add_argument('--iters', type=int, required=False, default=200000, help='Number of training steps. ')

    args = parser.parse_args()
    return args.__dict__

def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def make_model(config, env, action_noise_fun):
    model = None
    if config["algo_name"] == "TD3":
        model = TD3('MlpPolicy',
                    env=env,
                    gamma=config["gamma"],
                    learning_rate=config["learning_rate"],
                    buffer_size=config["buffer_size"],
                    learning_starts=config["learning_starts"],
                    train_freq=config["train_freq"],
                    gradient_steps=config["gradient_steps"],
                    batch_size=config["batch_size"],
                    tau=config["tau"],
                    policy_delay=config["policy_delay"],
                    action_noise=action_noise_fun,
                    target_policy_noise=config["target_policy_noise"],
                    target_noise_clip=config["target_noise_clip"],
                    verbose=config["verbose"],
                    tensorboard_log="./tb/{}/".format(config["session_ID"]),
                    policy_kwargs=config["policy_kwargs"])

    if config["algo_name"] == "A2C":
        model = A2C('MlpPolicy',
            env=env,
            gamma=config["gamma"],
            n_steps=config["n_steps"],
            vf_coef=config["vf_coef"],
            ent_coef = config["ent_coef"],
            max_grad_norm=config["max_grad_norm"],
            learning_rate=config["learning_rate"],
            alpha=config["alpha"],
            epsilon=config["epsilon"],
            lr_schedule=config["lr_schedule"],
            verbose=config["verbose"],
            tensorboard_log="./tb/{}/".format(config["session_ID"]),
            full_tensorboard_log=config["full_tensorboard_log"],
            policy_kwargs=dict(net_arch=[int(196), int(196)]))

    assert model is not None, "Alg name not found, exiting. "
    return model

def load_model(config):
    model = None
    if config["algo_name"] == "TD3":
        model = TD3.load("agents/{}".format(args["test_agent_path"]))
    if config["algo_name"] == "A2C":
        model = A2C.load("agents/{}".format(args["test_agent_path"]))
    if config["algo_name"] == "SAC":
        model = SAC.load("agents/{}".format(args["test_agent_path"]))
    if config["algo_name"] == "PPO2":
        model = PPO2.load("agents/{}".format(args["test_agent_path"]))
    assert model is not None, "Alg name not found, cannot load model, exiting. "
    return model

def make_action_noise_fun(config):
    return None

def test_agent(env, model, deterministic=True):
    for _ in range(100):
        obs = env.reset()
        cum_rew = 0
        while True:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            env.render()
            if done:
                print(cum_rew)
                break
    env.close()

if __name__ == "__main__":
    args = parse_args()
    algo_config = read_config(args["algo_config"])
    env_config = read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    for s in ["agents", "agents_cp", "tb"]:
        if not os.path.exists(s):
            os.makedirs(s)

    # Random ID of this session
    if config["default_session_ID"] is None:
        config["session_ID"] = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))
    else:
        config["session_ID"] = "TST"

    pprint(config)

    # Import correct env by name
    env_fun = my_utils.import_env(env_config["env_name"])

    if args["train"] or socket.gethostname() == "goedel":
        env = SubprocVecEnv([lambda : env_fun(config) for _ in range(config["n_envs"])], start_method='fork')
        model = make_model(config, env, None)

        checkpoint_callback = CheckpointCallback(save_freq=50000,
                                                 save_path='agents_cp/',
                                                 name_prefix=config["session_ID"], verbose=1)

        t1 = time.time()
        model.learn(total_timesteps=algo_config["iters"], callback=checkpoint_callback)
        t2 = time.time()

        print("Training time: {}".format(t2-t1))
        pprint(config)

        model.save("agents/{}_SB_policy".format(config["session_ID"]))
        env.close()

    if args["test"] and socket.gethostname() != "goedel":
        env = env_fun(config)

        if not args["train"]:
            model = load_model(config)

        test_agent(env, model, deterministic=True)