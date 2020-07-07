import os
import sys

import numpy as np
import torch as T
import src.my_utils as my_utils
import src.policies as policies
import random
import string
import socket
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def train(env, policy, params):
    policy_optim = T.optim.RMSprop(policy.parameters(), lr=params["policy_lr"], weight_decay=params["weight_decay"], eps=1e-8)

    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_new_states = []
    batch_terminals = []
    batch_step_counter = []

    batch_ctr = 0
    batch_rew = 0
    global_step_ctr = 0

    for i in range(params["iters"]):
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

            if params["animate"]:
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

        # If enough data gathered, then perform update
        if batch_ctr == params["batchsize"]:
            global_step_ctr += step_ctr

            batch_states = T.cat(batch_states)
            batch_new_states = T.cat(batch_new_states)
            batch_actions = T.cat(batch_actions)
            batch_rewards = T.cat(batch_rewards)

            # Scale rewards
            if params["normalize_rewards"]:
                batch_rewards = (batch_rewards - batch_rewards.mean()) / batch_rewards.std()

            # Calculate episode advantages
            batch_advantages = calc_advantages_MC(params["gamma"], batch_rewards, batch_terminals)

            if params["ppo_update_iters"] > 0:
                loss_policy = update_policy_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, params["ppo_update_iters"])
            else:
                loss_policy = update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages)

            print("Episode {}/{}, n_steps: {}, loss_policy: {}, mean ep_rew: {}".
                  format(i, params["iters"], global_step_ctr, loss_policy, batch_rew / params["batchsize"]))

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
                                "agents/{}_{}_{}_pg.p".format(env.__class__.__name__, policy.__class__.__name__, params["ID"]))
            T.save(policy.state_dict(), sdir)
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

        if False:
            # Symmetry loss
            batch_states_rev = batch_states.clone()

            # Joint angles
            batch_states_rev[:, 0:3] = batch_states[:, 3:6]
            batch_states_rev[:, 6:9] = batch_states[:, 9:12]
            batch_states_rev[:, 12:15] = batch_states[:, 15:18]

            batch_states_rev[:, 3:6] = batch_states[:, 0:3]
            batch_states_rev[:, 9:12] = batch_states[:, 6:9]
            batch_states_rev[:, 15:18] = batch_states[:, 12:15]
            batch_states_rev[:, 18] = batch_states[:, 18]
            batch_states_rev[:, 19] = -batch_states[:, 19]
            batch_states_rev[:, 20] = batch_states[:, 20]
            batch_states_rev[:, 21] = -batch_states[:, 21]

            if batch_states.shape[1] > 21:
                batch_states_rev[:, [22, 24, 26]] = batch_states[:, [23, 25, 27]]
                batch_states_rev[:, [23, 25, 27]] = batch_states[:, [22, 24, 26]]

            # Actions
            actions = policy(batch_states)
            actions_rev_pred = policy(batch_states_rev)
            actions_rev = T.zeros_like(actions)

            actions_rev[:, 0:3] = actions[:, 3:6]
            actions_rev[:, 6:9] = actions[:, 9:12]
            actions_rev[:, 12:15] = actions[:, 15:18]

            actions_rev[:, 3:6] = actions[:, 0:3]
            actions_rev[:, 9:12] = actions[:, 6:9]
            actions_rev[:, 15:18] = actions[:, 12:15]

            loss = (actions_rev_pred - actions_rev).pow(2).mean()
            print("Symmetry loss: {}".format(loss))
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

if __name__=="__main__":
    args = ["None", "flat", "straight"]
    if len(sys.argv) > 1:
        args = sys.argv

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 500000,
              "batchsize": 60,
              "max_steps": 80,
              "gamma": 0.97,
              "policy_lr": 0.0007,
              "weight_decay" : 0.0001,
              "ppo_update_iters" : 1,
              "normalize_rewards": False,
              "animate" : True,
              "train" : False,
              "note" : "Training: {}, {}".format(args[1], args[2]),
              "ID" : ID}

    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    #from src.envs.bullet_cartpole.cartpole.cartpole import CartPoleBulletEnv as env_fun
    #from src.envs.bullet_cartpole.hangpole_goal.hangpole_goal import HangPoleGoalBulletEnv as env_fun
    #from src.envs.bullet_cartpole.double_cartpole_goal.double_cartpole_goal import DoubleCartPoleBulletEnv as env_fun
    from src.envs.bullet_nexabot.hexapod.hexapod import HexapodBulletEnv as env_fun
    env = env_fun(animate=params["animate"], max_steps=params["max_steps"], step_counter=False, terrain_name=args[1], training_mode=args[2])

    # Test
    if params["train"]:
        print("Training")
        policy = policies.NN_PG(env, 96)
        print(params, env.obs_dim, env.act_dim, env.__class__.__name__, policy.__class__.__name__)
        train(env, policy, params)
    else:
        print("Testing")
        policy_name = "JUB" # 4B5 straight (try on real hex)
        policy_path = 'agents/{}_NN_PG_{}_pg.p'.format(env.__class__.__name__, policy_name)
        policy = policies.NN_PG(env, 96)
        policy.load_state_dict(T.load(policy_path))
        env.test(policy)
        print(policy_path)
