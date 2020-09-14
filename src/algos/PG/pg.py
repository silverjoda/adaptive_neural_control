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

def train(env, policy, config):
    policy_optim = T.optim.RMSprop(policy.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
                                   eps=1e-8, momentum=config["momentum"])

    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_new_states = []
    batch_terminals = []
    batch_step_counter = []

    batch_ctr = 0
    batch_rew = 0
    global_step_ctr = 0

    t1 = time.time()

    for i in range(config["iters"]):
        s_0 = env.reset()
        done = False
        step_ctr = 0

        while not done:
            batch_step_counter.append(step_ctr)

            # Sample action from policy
            action = policy.sample_action(my_utils.to_tensor(s_0, True)).detach()

            # Step action
            s_1, r, done, _ = env.step(action.squeeze(0).numpy())

            if abs(r) > 5:
                logging.warning("Warning! high reward ({})".format(r))

            step_ctr += 1

            # For calculating mean episode rew
            batch_rew += r

            if config["animate"]:
                env.render()

            # Record transition
            batch_states.append(my_utils.to_tensor(s_0, True))
            batch_actions.append(action)
            batch_rewards.append(my_utils.to_tensor(np.asarray(r, dtype=np.float32), True))
            batch_new_states.append(my_utils.to_tensor(s_1, True))
            batch_terminals.append(done)

            s_0 = s_1

        # Just completed an episode
        batch_ctr += 1
        global_step_ctr += step_ctr

        # If enough data gathered, then perform update
        if batch_ctr == config["batchsize"]:
            batch_states = T.cat(batch_states)
            batch_actions = T.cat(batch_actions)
            batch_rewards = T.cat(batch_rewards)

            # Scale rewards
            if config["normalize_rewards"]:
                batch_rewards = (batch_rewards - batch_rewards.mean()) / batch_rewards.std()

            # Calculate episode advantages
            batch_advantages = calc_advantages_MC(config["gamma"], batch_rewards, batch_terminals)

            if config["ppo_update_iters"] > 0:
                loss_policy = update_policy_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, config["ppo_update_iters"])
            else:
                loss_policy = update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages)
            t2 = time.time()
            print("Episode {}/{}, n_steps: {}, loss_policy: {}, mean ep_rew: {}, time per batch: {}".
                  format(i, config["iters"], global_step_ctr, loss_policy, batch_rew / config["batchsize"], t2 - t1))
            t1 = t2

            # Finally reset all batch lists
            batch_ctr = 0
            batch_rew = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_new_states = []
            batch_terminals = []
            batch_step_counter = []

        if i % 500 == 0 and i > 0:
            sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "agents/{}_{}_{}_pg.p".format(env.__class__.__name__, policy.__class__.__name__, config["ID"]))
            T.save(policy.state_dict(), sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, config))

def update_policy_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, update_iters):
    log_probs_old = policy.log_probs(batch_states, batch_actions).detach()
    c_eps = 0.2
    loss = None

    # Do ppo_update
    for k in range(update_iters):
        log_probs_new = policy.log_probs(batch_states, batch_actions)
        r = T.exp(log_probs_new - log_probs_old)
        loss = -T.mean(T.min(r * batch_advantages, r.clamp(1 - c_eps, 1 + c_eps) * batch_advantages))
        policy_optim.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(policy.parameters(), 0.7)
        policy_optim.step()

    return loss.data

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

def make_env(config, env_fun):
    def _init():
        env = env_fun(config)
        return env
    return _init

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

def test_agent(env, policy, deterministic=True):
    for _ in range(100):
        obs = env.reset()
        cum_rew = 0
        while True:
            action, _states = policy.forward(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            env.render()
            if done:
                print(cum_rew)
                break
    env.close()

if __name__=="__main__":
    args = parse_args()
    algo_config = read_config(args["algo_config"])
    env_config = read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    print(args)
    print(algo_config)
    print(env_config)

    # Import correct env by name
    env_fun = import_env(config["env_name"])
    env = env_fun(config)

    # Random ID of this session
    config["ID"] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

    policy = policies.NN_PG(env, config)

    if config["train"] or socket.gethostname() == "goedel":
        t1 = time.time()
        train(env, policy, config)
        t2 = time.time()

        print("Training time: {}".format(t2 - t1))
        print(args)
        print(algo_config)
        print(env_config)

    if config["test"] and socket.gethostname() != "goedel":
        env = make_env(config, env_fun)
        if not args["train"]:
            policy = policies.NN_PG(env, config)
            # policy = policies.RNN_PG(env, 18)
            policy.load_state_dict(T.load(config["test_agent"]))
        test_agent(env, policy, deterministic=True)


