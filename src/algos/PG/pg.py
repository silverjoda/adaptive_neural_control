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
import yaml
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
from torch.utils.tensorboard import SummaryWriter

def make_rollout(env, policy):
    #noisefun = my_utils.SimplexNoise(env.act_dim)
    obs = env.reset()
    observations = []
    clean_actions = []
    noisy_actions = []
    rewards = []
    while True:
        observations.append(obs)

        clean_act, noisy_act = policy.sample_action(my_utils.to_tensor(obs, True))
        clean_act = clean_act.squeeze(0).detach().numpy()
        noisy_act = noisy_act.squeeze(0).detach().numpy()
        #noisy_act = clean_act + noisefun()
        obs, r, done, _ = env.step(noisy_act)

        if abs(r) > 5:
            logging.warning("Warning! high reward ({})".format(r))

        if config["animate"]:
            env.render()

        clean_actions.append(clean_act)
        noisy_actions.append(noisy_act)
        rewards.append(r)
        if done: break
    terminals = [False] * len(observations)
    terminals[-1] = True
    return observations, clean_actions, noisy_actions, rewards, terminals

def train(env, policy, config):
    sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        f'agents/{config["session_ID"]}_pg_policy.p')

    policy_optim = None
    if config["policy_optim"] == "rmsprop":
        policy_optim = T.optim.RMSprop(policy.parameters(),
                                       lr=config["learning_rate"],
                                       weight_decay=config["weight_decay"],
                                       eps=1e-8, momentum=config["momentum"])
    if config["policy_optim"] == "sgd":
        policy_optim = T.optim.SGD(policy.parameters(),
                                   lr=config["learning_rate"],
                                   weight_decay=config["weight_decay"],
                                   momentum=config["momentum"])
    if config["policy_optim"] == "adam":
        policy_optim = T.optim.Adam(policy.parameters(),
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

    for i in range(config["iters"]):
        observations, clean_actions, actions, rewards, terminals = make_rollout(env, policy)

        batch_observations.extend(observations)
        batch_actions.extend(actions)
        batch_rewards.extend(rewards)
        batch_terminals.extend(terminals)

        # Just completed an episode
        batch_ctr += 1
        global_step_ctr += len(observations)

        # If enough data gathered, then perform update
        if batch_ctr == config["batchsize"]:
            batch_observations = T.from_numpy(np.array(batch_observations))
            batch_actions = T.from_numpy(np.array(batch_actions))
            batch_rewards = T.from_numpy(np.array(batch_rewards))

            # Log to TB
            if config["tb_writer"] is not None:
                # Pick random obs and make full activation log to tb
                rnd_obs = random.choice(observations)
                action_sample_full = policy.sample_action_w_activations(my_utils.to_tensor(rnd_obs, True))
                l1_activ, l1_normed, l1_nonlin, l2_activ, l2_normed, l2_nonlin, act, act_sampled = action_sample_full

                config["tb_writer"].add_histogram("Rnd_obs/L1_activ", l1_activ,
                                                  global_step=global_step_ctr)
                config["tb_writer"].add_histogram("Rnd_obs/L1_normed", l1_normed,
                                                  global_step=global_step_ctr)
                config["tb_writer"].add_histogram("Rnd_obs/L1_nonlin", l1_nonlin,
                                                  global_step=global_step_ctr)

                config["tb_writer"].add_histogram("Rnd_obs/L2_activ", l2_activ,
                                                  global_step=global_step_ctr)
                config["tb_writer"].add_histogram("Rnd_obs/L2_normed", l2_normed,
                                                  global_step=global_step_ctr)
                config["tb_writer"].add_histogram("Rnd_obs/L2_nonlin", l2_nonlin,
                                                  global_step=global_step_ctr)

                config["tb_writer"].add_histogram("Rnd_obs/Act", act,
                                                  global_step=global_step_ctr)
                config["tb_writer"].add_histogram("Rnd_obs/Act_sampled", act_sampled,
                                                  global_step=global_step_ctr)

                config["tb_writer"].add_scalar("Batch/Mean episode reward",
                                               batch_rewards.sum() / config["batchsize"],
                                               global_step=global_step_ctr)

                config["tb_writer"].add_histogram("Batch/Rewards", batch_rewards, global_step=global_step_ctr)
                config["tb_writer"].add_histogram("Batch/Observations", batch_observations, global_step=global_step_ctr)
                config["tb_writer"].add_histogram("Batch/Sampled actions", batch_actions, global_step=global_step_ctr)
                config["tb_writer"].add_scalar("Batch/Terminal step", len(batch_terminals) / config["batchsize"], global_step=global_step_ctr)

                for p in policy.named_parameters():
                    config["tb_writer"].add_histogram(f"Network/{p[0]}_param", p[1],
                                                      global_step=global_step_ctr)

            # Scale rewards
            if config["normalize_rewards"]:
                batch_rewards_for_advantages = (batch_rewards - batch_rewards.mean()) / batch_rewards.std()
            else:
                batch_rewards_for_advantages = batch_rewards

            # Calculate episode advantages
            batch_advantages = calc_advantages_MC(config["gamma"], batch_rewards_for_advantages, batch_terminals)

            if config["ppo_update_iters"] > 0:
                loss_policy = update_policy_ppo(policy, policy_optim, batch_observations, batch_actions, batch_advantages, config, global_step_ctr)
            else:
                loss_policy = update_policy(policy, policy_optim, batch_observations, batch_actions, batch_advantages, config, global_step_ctr)

            # Post update log
            if config["tb_writer"] is not None:
                config["tb_writer"].add_histogram("Batch/Advantages", batch_advantages, global_step=global_step_ctr)
                config["tb_writer"].add_scalar("Batch/Loss_policy", loss_policy, global_step=global_step_ctr)

            t2 = time.time()
            print("Episode {}/{}, n_steps: {}, loss_policy: {}, mean ep_rew: {}, time per batch: {}".
                  format(i, config["iters"], global_step_ctr, loss_policy, batch_rewards.sum() / config["batchsize"], t2 - t1))
            t1 = t2

            # Finally reset all batch lists
            batch_ctr = 0

            batch_observations = []
            batch_actions = []
            batch_rewards = []
            batch_terminals = []

            # Decay log_std
            policy.log_std -= config["log_std_decay"]

        if i % 500 == 0 and i > 0:
            T.save(policy.state_dict(), sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, config))

def update_policy_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, config, global_step_ctr):
    log_probs_old = policy.log_probs(batch_states, batch_actions).detach()
    loss = None

    # Do ppo_update
    for k in range(config["ppo_update_iters"]):
        log_probs_new = policy.log_probs(batch_states, batch_actions)
        r = T.exp(log_probs_new - log_probs_old)
        loss = -T.mean(T.min(r * batch_advantages, r.clamp(1 - config["c_eps"], 1 + config["c_eps"]) * batch_advantages))

        # Zero grads and backprop
        policy_optim.zero_grad()
        loss.backward()

        # Log gradients
        if config["tb_writer"] is not None:
            for p in policy.named_parameters():
                config["tb_writer"].add_histogram(f"Batch Grads/{p[0]}_grads_unclipped", p[1].grad,
                                                  global_step=global_step_ctr)

        #T.nn.utils.clip_grad_norm_(policy.parameters(), 0.7)

        # Log clipped gradients
        policy_optim.step()

    return loss.data

def update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages, config, global_step_ctr):
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
            print(action)
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


