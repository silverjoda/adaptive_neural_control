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
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize

def make_rollout(env, policy, config):
    obs = env.reset()
    observations = []
    actions = []
    rewards = []
    while True:
        observations.append(obs)

        act_np = policy.sample_action(obs)
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
    vf_optim = None
    if config["policy_optim"] == "rmsprop":
        policy_optim = T.optim.RMSprop(policy.parameters(),
                                       lr=config["policy_learning_rate"],
                                       weight_decay=config["weight_decay"],
                                       eps=1e-8, momentum=config["momentum"])
        vf_optim = T.optim.RMSprop(vf.parameters(),
                                       lr=config["vf_learning_rate"],
                                       weight_decay=config["weight_decay"],
                                       eps=1e-8, momentum=config["momentum"])
    if config["policy_optim"] == "sgd":
        policy_optim = T.optim.SGD(policy.parameters(),
                                   lr=config["policy_learning_rate"],
                                   weight_decay=config["weight_decay"],
                                   momentum=config["momentum"])
        vf_optim = T.optim.SGD(vf.parameters(),
                                   lr=config["vf_learning_rate"],
                                   weight_decay=config["weight_decay"],
                                   momentum=config["momentum"])
    if config["policy_optim"] == "adam":
        policy_optim = T.optim.Adam(policy.parameters(),
                                    lr=config["policy_learning_rate"],
                                    weight_decay=config["weight_decay"])
        vf_optim = T.optim.Adam(vf.parameters(),
                                    lr=config["vf_learning_rate"],
                                    weight_decay=config["weight_decay"])
    assert policy_optim is not None

    batch_observations = []
    batch_actions = []
    batch_rewards = []
    batch_terminals = []

    batch_ctr = 0
    global_batch_ctr = 0
    global_step_ctr = 0

    t1 = time.time()

    while global_step_ctr < config["n_total_steps_train"]:
        observations, actions, rewards, terminals = make_rollout(env, policy, config)

        batch_observations.extend(observations)
        batch_actions.extend(actions)
        batch_rewards.extend(rewards)
        batch_terminals.extend(terminals)

        # Just completed an episode
        batch_ctr += 1
        global_batch_ctr += 1
        global_step_ctr += len(observations)

        # If enough data gathered, then perform update
        if batch_ctr == config["batchsize"]:
            batch_observations = T.from_numpy(np.array(batch_observations))
            batch_actions = T.from_numpy(np.array(batch_actions))
            batch_rewards = T.from_numpy(np.array(batch_rewards))

            # Calculate episode advantages
            #batch_advantages = calc_advantages_MC(config, batch_rewards, batch_terminals)
            batch_advantages = calc_advantages(config["gamma"], vf, batch_observations, batch_rewards, batch_terminals)
            loss_policy = update_policy(policy, policy_optim, batch_observations, batch_actions, batch_advantages.detach())
            loss_vf = update_vf(vf_optim, batch_advantages)

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
            print("N_total_steps_train {}/{}, loss_policy: {}, loss_vf: {}, mean ep_rew: {}, time per batch: {}, mean_episode_len".
                  format(global_step_ctr,
                         config["n_total_steps_train"],
                         loss_policy,
                         loss_vf,
                         batch_rewards.sum() / config["batchsize"],
                         t2 - t1),
                         len(batch_rewards) / config["batchsize"])
            t1 = t2

            # Finally reset all batch lists
            batch_ctr = 0

            batch_observations = []
            batch_actions = []
            batch_rewards = []
            batch_terminals = []

            # Decay log_std
            #policy.log_std -= config["log_std_decay"]

        if global_batch_ctr % 1000 == 0 and global_batch_ctr > 0 and config["save_policy"]:
            T.save(policy.state_dict(), sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, config))

    if config["save_policy"]:
        T.save(policy.state_dict(), sdir)
        print("Finished training, saved policy at {} with params {}".format(sdir, config))

def update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages):
    # Get action log probabilities
    log_probs = policy.log_probs(batch_states, batch_actions)

    # Calculate loss function
    loss = -T.mean(log_probs * batch_advantages)

    # Backward pass on policy
    policy_optim.zero_grad()
    loss.backward()

    # Step policy update
    policy_optim.step()

    return loss.data

def update_vf(vf_optim, batch_advantages):
    vf_optim.zero_grad()
    loss = T.mean(0.5 * T.pow(batch_advantages, 2))
    loss.backward()
    vf_optim.step()
    return loss.data.detach()

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

def calc_advantages(gamma, vf, batch_observations, batch_rewards, batch_terminals):
    batch_values = vf(batch_observations)
    targets = []
    for i in reversed(range(len(batch_rewards))):
        r, v, t = batch_rewards[i], batch_values[i], batch_terminals[i]
        if t:
            R = r - v
        else:
            v_next = batch_values[i + 1]
            R = r + gamma * v_next - v
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
    parser.add_argument('--algo_config', type=str, default="configs/ac_default_config.yaml", required=False,
                        help='Algorithm config file name. ')
    parser.add_argument('--env_config', type=str, default="hexapod_config.yaml", required=False,
                        help='Env config file name. ')
    parser.add_argument('--iters', type=int, required=False, default=200000, help='Number of training steps. ')

    args = parser.parse_args()
    return args.__dict__

def test_agent(env, policy, N=100, print_rew=False, render=True):
    total_rew = 0
    for i in range(N):
        obs = env.reset()
        episode_rew = 0
        while True:
            action = policy.sample_action(obs)
            obs, reward, done, info = env.step(action)
            episode_rew += reward
            total_rew += reward
            if render:
                env.render()
            if done:
                if print_rew:
                    print(episode_rew)
                break
    return total_rew

def setup_train(config):
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
    # env = VecNormalize(env_fun(config))
    # env = SubprocVecEnv([lambda: env_fun(config) for _ in range(1)], start_method='fork')

    policy = my_utils.make_policy(env, config)
    vf = my_utils.make_vf(env, config)

    config["tb_writer"] = None
    if config["log_tb"]:
        tb_writer = SummaryWriter(f'tb/{config["session_ID"]}')
        config["tb_writer"] = tb_writer

    return env, policy, vf

if __name__=="__main__":
    args = parse_args()
    algo_config = my_utils.read_config(args["algo_config"])
    env_config = my_utils.read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    print(config)

    env, policy, vf = setup_train(config)

    if config["train"] or socket.gethostname() == "goedel":
        t1 = time.time()
        train(env, policy, vf, config)
        t2 = time.time()

        print("Training time: {}".format(t2 - t1))
        print(config)

    if config["test"] and socket.gethostname() != "goedel":
        if not args["train"]:
            policy.load_state_dict(T.load(config["test_agent_path"]))
        test_agent(env, policy, print_rew=True)


