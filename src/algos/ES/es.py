import cma
import logging
import os
import random
import socket
import string
import sys
import time
import argparse

import cma
import torch
import torch as T
import yaml
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

import src.my_utils as my_utils
import src.policies as policies

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
from torch.utils.tensorboard import SummaryWriter

def f_wrapper(env, policy):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        vector_to_parameters(torch.from_numpy(w).float(), policy.parameters())

        while not done:
            with torch.no_grad():
                act = policy(my_utils.to_tensor(obs, True))
            obs, rew, done, _ = env.step(act.squeeze(0).numpy())
            reward += rew

        return -reward
    return f

def train(env, policy, config):
    w = parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, config["cma_std"])
    f = f_wrapper(env, policy)

    sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        f'agents/{config["session_ID"]}_es_policy.p')

    print(f'N_params: {len(w)}')

    it = 0
    try:
        while not es.stop():
            if config["tb_writer"] is not None:
                for p in policy.named_parameters():
                    config["tb_writer"].add_histogram(f"Network/{p[0]}_param", p[1],
                                                      global_step=it)

            it += 1
            if it > config["iters"]:
                break
            X = es.ask()
            es.tell(X, [f(x) for x in X])
            es.disp()

    except KeyboardInterrupt:
        print("User interrupted process.")

    vector_to_parameters(torch.from_numpy(es.result.xbest).float(), policy.parameters())
    T.save(policy.state_dict(), sdir)
    print("Saved agent, {}".format(sdir))

    return es.result.fbest

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
    parser.add_argument('--algo_config', type=str, default="configs/pg_default_config.yaml", required=False,
                        help='Algorithm config file name. ')
    parser.add_argument('--env_config', type=str, default="hexapod_config.yaml", required=False,
                        help='Env config file name. ')
    parser.add_argument('--iters', type=int, required=False, default=200000, help='Number of training steps. ')

    args = parser.parse_args()
    return args.__dict__

def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def test_agent(env, policy):
    for _ in range(100):
        obs = env.reset()
        cum_rew = 0
        while True:
            action = policy(my_utils.to_tensor(obs, True)).detach().squeeze(0).numpy()
            obs, reward, done, info = env.step(action)
            cum_rew += reward

            if done:
                print(cum_rew)
                break
    env.close()

if __name__=="__main__":
    args = parse_args()
    algo_config = read_config(args["algo_config"])
    env_config = read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    print(config)

    for s in ["agents", "agents_cp", "tb"]:
        if not os.path.exists(s):
            os.makedirs(s)

    # Random ID of this session
    if config["default_session_ID"] is None:
        config["session_ID"] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    else:
        config["session_ID"] = "TST"

    # Import correct env by name
    env_fun = my_utils.import_env(config["env_name"])
    env = env_fun(config)

    policy = my_utils.make_policy(env, config)

    if config["log_tb_all"]:
        tb_writer = SummaryWriter(f'tb/{config["session_ID"]}')
        config["tb_writer"] = tb_writer
    else:
        config["tb_writer"] = None

    if config["train"] or socket.gethostname() == "goedel":
        t1 = time.time()
        train(env, policy, config)
        t2 = time.time()

        print("Training time: {}".format(t2 - t1))
        print(config)

    if config["test"] and socket.gethostname() != "goedel":
        if not args["train"]:
            policy.load_state_dict(T.load(config["test_agent_path"]))
        test_agent(env, policy)