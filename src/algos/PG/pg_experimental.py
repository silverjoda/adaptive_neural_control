import os
import sys

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import time
import src.my_utils as my_utils
import src.policies as policies
import random
import string
import socket
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def train(env, policy, valuefun, params):
    policy_optim = T.optim.RMSprop(policy.parameters(), lr=params["policy_lr"], weight_decay=params["weight_decay"], eps=1e-4)
    valuefun_optim = T.optim.RMSprop(valuefun.parameters(), lr=params["valuefun_lr"], weight_decay=params["weight_decay"], eps=1e-4)

    batch_states = []
    batch_states_v = []
    batch_actions = []
    batch_rewards = []
    batch_new_states = []
    batch_new_states_v = []
    batch_terminals = []

    batch_ctr = 0
    batch_rew = 0
    global_step_ctr = 0

    loss_v = None

    def step_to_encoding(t):
        return (float(t) / params["max_steps"]) * 2 - 1

    for i in range(params["iters"]):
        s_0 = env.reset()
        s_0_v = np.concatenate((s_0, [step_to_encoding(0)]))
        done = False
        step_ctr = 0

        while not done:
            # Sample action from policy
            action = policy.sample_action(my_utils.to_tensor(s_0, True)).detach()

            # Step action
            s_1, r, done, _ = env.step(action.squeeze(0).numpy())
            s_1_v = np.concatenate((s_1, [step_to_encoding(step_ctr + 1)]))

            if abs(r) > 5:
                logging.warning("Warning! high reward ({})".format(r))

            step_ctr += 1

            # For calculating mean episode rew
            batch_rew += r

            if params["animate"]:
                env.render()

            # Record transition
            batch_states.append(my_utils.to_tensor(s_0, True))
            batch_states_v.append(my_utils.to_tensor(s_0_v, True))
            batch_actions.append(action)
            batch_rewards.append(my_utils.to_tensor(np.asarray(r, dtype=np.float32), True))
            batch_new_states.append(my_utils.to_tensor(s_1, True))
            batch_new_states_v.append(my_utils.to_tensor(s_1_v, True))
            batch_terminals.append(done)

            s_0 = s_1
            s_0_v = s_1_v

        # Just completed an episode
        batch_ctr += 1

        # If enough data gathered, then perform update
        if batch_ctr == params["batchsize"]:
            global_step_ctr += step_ctr

            batch_states = T.cat(batch_states)
            batch_states_v = T.cat(batch_states_v)
            batch_new_states = T.cat(batch_new_states)
            batch_new_states_v = T.cat(batch_new_states_v)
            batch_actions = T.cat(batch_actions)
            batch_rewards = T.cat(batch_rewards)

            # Scale rewards
            #batch_rewards = (batch_rewards - batch_rewards.mean()) / batch_rewards.std()

            # Calculate episode advantages
            batch_targets, batch_advantages = calc_advantages(valuefun, params["gamma"], batch_states_v, batch_rewards, batch_new_states_v, batch_terminals)

            #loss_policy = update_policy_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, params["ppo_update_iters"])
            loss_policy = update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages)
            loss_v = update_V(valuefun, valuefun_optim, batch_states_v, batch_rewards, batch_terminals, batch_targets)

            print("Episode {}/{}, n_steps: {}, loss_V: {}, loss_policy: {}, mean ep_rew: {}".
                  format(i, params["iters"], global_step_ctr, loss_v, loss_policy, batch_rew / params["batchsize"]))

            # Finally reset all batch lists
            batch_ctr = 0
            batch_rew = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_new_states = []
            batch_terminals = []

        if i % 500 == 0 and i > 0:
            sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "agents/{}_{}_{}_pg.p".format(env.__class__.__name__, policy.__class__.__name__, params["ID"]))
            T.save(policy, sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, params))


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
        policy.soft_clip_grads(3.)
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

def update_V(V, V_optim, batch_states, batch_rewards, batch_terminals, targets):
    assert len(batch_states) == len(batch_rewards) == len(batch_terminals)
    Vs = V(batch_states)

    V_optim.zero_grad()
    loss = F.mse_loss(Vs, targets)
    loss.backward()
    V_optim.step()
    return loss.data

def calc_advantages(V, gamma, batch_states, batch_rewards, batch_new_states, batch_terminals):
    with T.no_grad():
        Vs = V(batch_states)
        Vs_ = V(batch_new_states)
        advantages = []
        targets = []
        for r, t, vs, vs_ in zip(batch_rewards, batch_terminals, Vs, Vs_):
            target = r + (1.0 - t) * gamma * vs_
            targets.append(target)
            advantages.append(target - vs)
        return T.cat(targets), T.cat(advantages)

if __name__=="__main__":
    #T.set_num_threads(1)

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 500000,
              "batchsize" : 60,
              "max_steps" : 150,
              "gamma" : 0.995,
              "policy_lr": 0.001,
              "valuefun_lr": 0.003,
              "weight_decay" : 0.0001,
              "ppo_update_iters" : 1,
              "animate" : False,
              "train" : True,
              "note" : "...",
              "ID" : ID}

    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    #from src.envs.bullet_cartpole.cartpole.cartpole import CartPoleBulletEnv as env_fun
    from src.envs.bullet_cartpole.hangpole_goal.hangpole_goal import HangPoleGoalBulletEnv as env_fun
    #from src.envs.bullet_cartpole.double_cartpole_goal.double_cartpole_goal import DoubleCartPoleBulletEnv as env_fun
    #from src.envs.bullet_nexabot.quadruped.quadruped import QuadrupedBulletEnv as env_fun
    env = env_fun(animate=params["animate"], max_steps=params["max_steps"])

    # Test
    if params["train"]:
        print("Training")
        policy = policies.NN_PG(env, 24)
        valuefun = policies.NN_PG_VF(env, 24)
        print(params, env.obs_dim, env.act_dim, env.__class__.__name__, policy.__class__.__name__)
        train(env, policy, valuefun, params)
    else:
        print("Testing")
        policy_name = "9R7" # 660 hangpole (15minstraining)
        policy_path = 'agents/{}_NN_PG_{}_pg.p'.format(env.__class__.__name__, policy_name)
        policy = T.load(policy_path)

        env.test(policy)
        print(policy_path)








