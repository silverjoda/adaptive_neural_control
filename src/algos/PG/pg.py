import os
import sys
import time
import numpy as np
import torch as T
import src.my_utils as my_utils
import src.policies as policies
import random
import string
import socket
import argparse
import yaml
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
from torch.utils.tensorboard import SummaryWriter

def make_rollout(env, policy):
    obs = env.reset()
    observations = []
    next_observations = []
    clean_actions = []
    noisy_actions = []
    rewards = []
    step_ctr_list = []
    episode_rew = 0
    step_ctr = 0
    while True:
        step_ctr_list.append(step_ctr)
        observations.append(obs)

        clean_act, noisy_act = policy.sample_action(my_utils.to_tensor(obs, True)).squeeze(0).detach().numpy()
        obs, r, done, _ = env.step(noisy_act)

        if abs(r) > 5:
            logging.warning("Warning! high reward ({})".format(r))

        step_ctr += 1
        episode_rew += r

        if config["animate"]:
            env.render()

        clean_actions.append(clean_act)
        noisy_actions.append(noisy_act)
        rewards.append(r)
        next_observations.append(obs)
        if done: break
    terminals = [False] * len(observations)
    terminals[-1] = True
    return observations, next_observations, clean_actions, noisy_actions, rewards, terminals, step_ctr_list

def train(env, policy, config):
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

    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_terminals = []

    batch_ctr = 0
    global_step_ctr = 0

    t1 = time.time()

    for i in range(config["iters"]):
        observations, next_observations, clean_actions, actions, rewards, terminals, step_ctr_list = make_rollout(env, policy)

        if config["tb_writer"] is not None:
            # Pick random obs and make full activation log to tb
            rnd_obs = random.choice(observations)
            l1, l2, clean_act, noisy_act = policy.sample_action_w_activations(my_utils.to_tensor(rnd_obs, True)).squeeze(0).detach().numpy()

            config["tb_writer"].add_histogram("L1_activations", l1, global_step=global_step_ctr)
            config["tb_writer"].add_histogram("L2_activations", l2, global_step=global_step_ctr)
            config["tb_writer"].add_histogram("action_activations", clean_act, global_step=global_step_ctr)
            config["tb_writer"].add_histogram("noisy_action_activations", noisy_act, global_step=global_step_ctr)

        # Log to tb
        if config["tb_writer"] is not None:
            config["tb_writer"].add_scalar("Rewards_sum")
            config["tb_writer"].add_histogram("Rewards histogram", rewards, global_step=global_step_ctr)
            config["tb_writer"].add_histogram("Observations", observations, global_step=global_step_ctr)
            config["tb_writer"].add_histogram("Clean actions", clean_actions, global_step=global_step_ctr)
            config["tb_writer"].add_histogram("Noisy actions", actions, global_step=global_step_ctr)
            config["tb_writer"].add_scalar("Terminal step", len(terminals), global_step=global_step_ctr)

        batch_states.extend(observations)
        batch_actions.extend(actions)
        batch_rewards.extend(rewards)
        batch_terminals.extend(terminals)

        # Just completed an episode
        batch_ctr += 1
        global_step_ctr += len(observations)

        # If enough data gathered, then perform update
        if batch_ctr == config["batchsize"]:
            batch_states = T.from_numpy(np.array(batch_states))
            batch_actions = T.from_numpy(np.array(batch_actions))
            batch_rewards = T.from_numpy(np.array(batch_rewards))

            if config["tb_writer"] is not None:
                config["tb_writer"].add_histogram("Batch rewards histogram", batch_rewards, global_step=global_step_ctr)

            # Scale rewards
            if config["normalize_rewards"]:
                batch_rewards_normalized = (batch_rewards - batch_rewards.mean()) / batch_rewards.std()
                batch_rewards_for_advantages = batch_rewards_normalized
            else:
                batch_rewards_for_advantages = batch_rewards

            # Calculate episode advantages
            batch_advantages = calc_advantages_MC(config["gamma"],
                                                  batch_rewards_for_advantages,
                                                  batch_terminals)

            if config["tb_writer"] is not None:
                config["tb_writer"].add_histogram("Batch advantages histogram", batch_advantages, global_step=global_step_ctr)

            if config["ppo_update_iters"] > 0:
                loss_policy = update_policy_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, config["ppo_update_iters"], config)
            else:
                loss_policy = update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages, config)

            # Post update log
            if config["tb_writer"] is not None:
                config["tb_writer"].add_scalar(loss_policy)
                config["tb_writer"].add_scalar(batch_rewards / config["batchsize"])

                for p in policy.named_parameters():
                    config["tb_writer"].add_histogram(f"{p[0]}_param", p[1],
                                                      global_step=global_step_ctr)

            t2 = time.time()
            print("Episode {}/{}, n_steps: {}, loss_policy: {}, mean ep_rew: {}, time per batch: {}".
                  format(i, config["iters"], global_step_ctr, loss_policy, batch_rewards / config["batchsize"], t2 - t1))
            t1 = t2

            # Finally reset all batch lists
            batch_ctr = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_terminals = []

        if i % 500 == 0 and i > 0:
            sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "agents/{}_{}_{}_pg.p".format(env.__class__.__name__, policy.__class__.__name__, config["ID"]))
            T.save(policy.state_dict(), sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, config))

def update_policy_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, update_iters, config, global_step_ctr):
    log_probs_old = policy.log_probs(batch_states, batch_actions).detach()
    loss = None

    # Do ppo_update
    for k in range(update_iters):
        log_probs_new = policy.log_probs(batch_states, batch_actions)
        r = T.exp(log_probs_new - log_probs_old)
        loss = -T.mean(T.min(r * batch_advantages, r.clamp(1 - config["c_eps"], 1 + config["c_eps"]) * batch_advantages))

        # Log activations in each layer
        for p in policy.parameters():
            config["tb_writer"].add_histogram("Batch advantages histogram", batch_advantages,
                                              global_step=global_step_ctr)

        # Zero grads and backprop
        policy_optim.zero_grad()
        loss.backward()

        # Log gradients
        for p in policy.named_parameters():
            config["tb_writer"].add_histogram(f"{p[0]}_grads_unclipped", p[1].grad(),
                                              global_step=global_step_ctr)

        T.nn.utils.clip_grad_norm_(policy.parameters(), 0.7)

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
    policy_optim.step()

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
                        help='Algorithm config flie name. ')
    parser.add_argument('--env_config', type=str, default="hexapod_config.yaml", required=False,
                        help='Env config file name. ')
    parser.add_argument('--iters', type=int, required=False, default=200000, help='Number of training steps. ')

    args = parser.parse_args()
    return args.__dict__

def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def import_env(name):
    if name == "hexapod":
        from src.envs.bullet_nexabot.hexapod.hexapod import HexapodBulletEnv as env_fun
    elif name == "quadrotor":
        from src.envs.bullet_quadrotor.quadrotor import QuadrotorBulletEnv as env_fun
    elif name == "buggy":
        from src.envs.bullet_buggy.buggy import BuggyBulletEnv as env_fun
    elif name == "quadruped":
        from src.envs.bullet_nexabot.quadruped.quadruped import QuadrupedBulletEnv as env_fun
    else:
        raise TypeError
    return env_fun

def make_action_noise_fun(config):
    return None

def test_agent(env, policy):
    for _ in range(100):
        obs = env.reset()
        cum_rew = 0
        while True:
            action = policy.sample_action(my_utils.to_tensor(obs, True)).detach().squeeze(0).numpy()
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            env.render()
            if done:
                print(cum_rew)
                break
    env.close()

def make_policy(env, config):
    if config["policy_type"] == "slp":
        return policies.SLP_PG(env, config)
    elif config["policy_type"] == "mlp":
        return policies.NN_PG(env, config)
    elif config["policy_type"] == "mlp_def":
        return policies.NN_PG_DEF(env, config)
    elif config["policy_type"] == "rnn":
        return policies.RNN_PG(env, config)
    else:
        raise TypeError

if __name__=="__main__":
    args = parse_args()
    algo_config = read_config(args["algo_config"])
    env_config = read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    print(config)

    # Import correct env by name
    env_fun = import_env(config["env_name"])
    env = env_fun(config)

    # Random ID of this session
    config["ID"] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    policy = make_policy(env, config)

    tb_writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    config["tb_writer"] = tb_writer

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


