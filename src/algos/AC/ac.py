import os
import sys
import time
import numpy as np
import torch as T
import src.my_utils as my_utils
import random
import string
import socket
import argparse
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
from torch.utils.tensorboard import SummaryWriter

def make_rollout(env, policy):
    obs = env.reset()
    observations = []
    actions = []
    rewards = []
    while True:
        observations.append(obs)

        act_np = policy.sample_action(my_utils.to_tensor(obs, True))
        obs, r, done, _ = env.step(act_np)

        if abs(r) > 5:
            logging.warning("Warning! high reward ({})".format(r))

        if config["animate"]:
            env.render()

        actions.append(act_np)
        rewards.append(r)
        if done: break
    terminals = [False] * len(observations)
    terminals[-1] = True
    return observations, actions, rewards, terminals

def train(env, policy, vf, config):
    sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        f'agents/{config["session_ID"]}_AC_policy.p')

    policy_optim = None
    if config["policy_optim"] == "rmsprop":
        policy_optim = T.optim.RMSprop(policy.parameters(),
                                       lr=config["learning_rate"],
                                       weight_decay=config["weight_decay"],
                                       eps=1e-8, momentum=config["momentum"])
        vf_optim = T.optim.RMSprop(vf.parameters(),
                                       lr=config["learning_rate"],
                                       weight_decay=config["weight_decay"],
                                       eps=1e-8, momentum=config["momentum"])
    if config["policy_optim"] == "sgd":
        policy_optim = T.optim.SGD(policy.parameters(),
                                   lr=config["learning_rate"],
                                   weight_decay=config["weight_decay"],
                                   momentum=config["momentum"])
        vf_optim = T.optim.SGD(vf.parameters(),
                                   lr=config["learning_rate"],
                                   weight_decay=config["weight_decay"],
                                   momentum=config["momentum"])
    if config["policy_optim"] == "adam":
        policy_optim = T.optim.Adam(policy.parameters(),
                                    lr=config["learning_rate"],
                                    weight_decay=config["weight_decay"])
        vf_optim = T.optim.Adam(vf.parameters(),
                                    lr=config["learning_rate"],
                                    weight_decay=config["weight_decay"])
    assert policy_optim is not None

    batch_observations = []
    batch_actions = []
    batch_rewards = []
    batch_terminals = []

    batch_ctr = 0
    global_step_ctr = 0

    t1 = time.time()

    while global_step_ctr < config["n_total_steps_train"]:
        observations, actions, rewards, terminals = make_rollout(env, policy)

        batch_observations.append(observations)
        batch_actions.append(actions)
        batch_rewards.append(rewards)
        batch_terminals.append(terminals)

        # Just completed an episode
        batch_ctr += 1
        global_step_ctr += len(observations)

        # If enough data gathered, then perform update
        if batch_ctr == config["batchsize"]:
            batch_observations = T.from_numpy(np.array(batch_observations))
            batch_actions = T.from_numpy(np.array(batch_actions))
            batch_rewards = T.from_numpy(np.array(batch_rewards))

            # Calculate episode advantages
            #batch_advantages = calc_advantages_MC(config, batch_rewards, batch_terminals)
            batch_advantages = calc_advantages(config, vf, batch_rewards, batch_terminals)
            loss_policy = update_policy(policy, policy_optim, batch_observations, batch_actions, batch_advantages)
            loss_vf = update_vf(vf, policy_optim, batch_observations, batch_actions, batch_advantages)

            # Post update log
            if config["tb_writer"] is not None:
                config["tb_writer"].add_histogram("Batch/Advantages", batch_advantages, global_step=global_step_ctr)
                config["tb_writer"].add_scalar("Batch/Loss_policy", loss_policy, global_step=global_step_ctr)

                config["tb_writer"].add_histogram("Batch/Rewards", batch_rewards, global_step=global_step_ctr)
                config["tb_writer"].add_histogram("Batch/Observations", batch_observations,
                                                  global_step=global_step_ctr)
                config["tb_writer"].add_histogram("Batch/Sampled actions", batch_actions,
                                                  global_step=global_step_ctr)
                config["tb_writer"].add_scalar("Batch/Terminal step", len(batch_terminals) / config["batchsize"],
                                               global_step=global_step_ctr)

            t2 = time.time()
            print("N_total_steps_train {}/{}, loss_policy: {}, mean ep_rew: {}, time per batch: {}".
                  format(global_step_ctr, config["n_total_steps_train"], loss_policy, batch_rewards.sum() / config["batchsize"], t2 - t1))
            t1 = t2

            # Finally reset all batch lists
            batch_ctr = 0

            batch_observations = []
            batch_actions = []
            batch_rewards = []
            batch_terminals = []

            # Decay log_std
            policy.log_std -= config["log_std_decay"]

        if global_step_ctr % 500000 < config["batchsize"] * config["max_steps"] and global_step_ctr > 0:
            T.save(policy.state_dict(), sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, config))

def update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages):
    # Get action log probabilities
    log_probs = policy.log_probs(batch_states, batch_actions)

    # Calculate loss function
    loss = -T.mean(log_probs * batch_advantages)

    # Backward pass on policy
    policy_optim.zero_grad()
    loss.backward()

    # Step policy update
    policy_optim.step()#

    return loss.data

def update_vf(vf, vf_optim, batch_states, batch_advantages):
    pass

def calc_advantages_MC(gamma, batch_rewards, batch_terminals):
    # Monte carlo estimate of targets
    targets = []
    with T.no_grad():
        for r, t in zip(reversed(batch_rewards), reversed(batch_terminals)):
            if t:
                R = r
            else:
                R = r + gamma * R
            targets.append(R.view(1, 1))
        targets = T.cat(list(reversed(targets)))
    return targets

def calc_advantages(gamma, vf, batch_rewards, batch_terminals):
    # Monte carlo estimate of targets
    targets = []
    for r, t in zip(reversed(batch_rewards), reversed(batch_terminals)):
        if t:
            R = r
        else:
            R = r + gamma * R
        targets.append(R.view(1, 1))
    targets = T.cat(list(reversed(targets)))
    return targets

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

def test_agent(env, policy):
    #env.test_agent(policy)
    #exit()
    for _ in range(100):
        obs = env.reset()
        cum_rew = 0
        while True:
            action, noisy_action = policy.sample_action(my_utils.to_tensor(obs, True))
            obs, reward, done, info = env.step(action.detach().squeeze(0).numpy())
            cum_rew += reward
            env.render()
            if done:
                print(cum_rew)
                break
    env.close()

if __name__=="__main__":
    args = parse_args()
    algo_config = my_utils.read_config(args["algo_config"])
    env_config = my_utils.read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    print(config)

    for s in ["agents", "agents_cp", "tb"]:
        if not os.path.exists(s):
            os.makedirs(s)

    # Random ID of this session
    if config["default_session_ID"] is None:
        config["session_ID"] = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))
    else:
        config["session_ID"] = "TST"

    # Import correct env by name
    env_fun = my_utils.import_env(config["env_name"])
    env = env_fun(config)

    policy = my_utils.make_policy(env, config)
    vf = my_utils.make_vf(env, config)

    if config["log_tb"]:
        tb_writer = SummaryWriter(f'tb/{config["session_ID"]}')
        config["tb_writer"] = tb_writer

    if config["train"] or socket.gethostname() == "goedel":
        t1 = time.time()
        train(env, policy, vf, config)
        t2 = time.time()

        print("Training time: {}".format(t2 - t1))
        print(config)

    if config["test"] and socket.gethostname() != "goedel":
        if not args["train"]:
            policy.load_state_dict(T.load(config["test_agent_path"]))
        test_agent(env, policy)


