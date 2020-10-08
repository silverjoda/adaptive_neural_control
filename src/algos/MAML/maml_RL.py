import argparse
import logging
import os
import random
import socket
import string
import sys
import time
from copy import deepcopy

import numpy as np
import torch as T

import src.my_utils as my_utils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description='Pass in parameters. ')
    parser.add_argument('--train', action='store_true', required=False,
                        help='Flag indicating whether the training process is to be run. ')
    parser.add_argument('--test', action='store_true', required=False,
                        help='Flag indicating whether the testing process is to be run. ')
    parser.add_argument('--animate', action='store_true', required=False,
                        help='Flag indicating whether the environment will be rendered. ')
    parser.add_argument('--test_agent_path', type=str, default=".", required=False,
                        help='Path of test agent. ')
    parser.add_argument('--algo_config', type=str, default="configs/maml_default_config.yaml", required=False,
                        help='Algorithm config flie name. ')
    parser.add_argument('--env_config', type=str, required=True,
                        help='Env config file name. ')
    parser.add_argument('--n_meta_iters', type=int, required=False, default=10000, help='Number of meta training steps. ')

    args = parser.parse_args()
    return args.__dict__

class MAMLRLTrainer:
    def __init__(self, env, policy, config):
        self.env = env
        self.policy = policy
        self.config = config

    def make_rollout(self, policy):
        self.env.set_randomize_env(False)
        obs = self.env.reset()
        observations = []
        clean_actions = []
        noisy_actions = []
        rewards = []
        while True:
            observations.append(obs)
            clean_act, noisy_act = policy.sample_action(my_utils.to_tensor(obs, True))
            clean_act = clean_act.squeeze(0).detach().numpy()
            noisy_act = noisy_act.squeeze(0).detach().numpy()
            obs, r, done, _ = self.env.step(noisy_act)
            clean_actions.append(clean_act)
            noisy_actions.append(noisy_act)
            rewards.append(r)
            if done: break
        terminals = [False] * len(observations)
        terminals[-1] = True
        return observations, clean_actions, noisy_actions, rewards, terminals

    def update_policy(self, policy, policy_optim, batch_states, batch_actions, batch_advantages, create_graph=False):

        # Get action log probabilities
        log_probs = policy.log_probs(batch_states, batch_actions)

        # Calculate loss function
        loss = -T.mean(log_probs * batch_advantages)

        # Backward pass on policy
        policy_optim.zero_grad()
        loss.backward(create_graph=create_graph)

        # Step policy update
        policy_optim.step()

        return loss.data.detach()

    def calc_advantages_MC(self, gamma, batch_rewards, batch_terminals):
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

    def comp_gradient_on_env_sample(self, policy):
        # Returns gradient with respect to meta parameters.

        # Make copy policy from meta params
        policy_copy = deepcopy(policy)
        for p_c, p in zip(policy_copy.parameters(), policy.parameters()):
            p_c.data = T.clone(p.data)

        # Make optimizer for this policy
        policy_copy_opt = T.optim.SGD(policy_copy.parameters(),
                              lr=self.config["learning_rate"],
                              momentum=self.config["momentum"],
                              weight_decay=self.config["w_decay"])

        # Randomize environment
        self.env.set_randomize_env(True)
        self.env.reset()

        # Do k updates
        trn_loss = 0
        for _ in range(self.config["k"]):
            # Perform rollouts
            batch_observations = []
            batch_actions = []
            batch_rewards = []
            batch_terminals = []

            for i in range(self.config["batchsize"]):
                observations, clean_actions, noisy_actions, rewards, terminals = self.make_rollout(policy_copy)

                batch_observations.extend(observations)
                batch_actions.extend(noisy_actions)
                batch_rewards.extend(rewards)
                batch_terminals.extend(terminals)

            batch_observations = T.from_numpy(np.array(batch_observations))
            batch_actions = T.from_numpy(np.array(batch_actions))
            batch_rewards = T.from_numpy(np.array(batch_rewards))
            batch_terminals = T.from_numpy(np.array(batch_terminals))

            batch_advantages = self.calc_advantages_MC(self.config["gamma"], batch_rewards, batch_terminals)
            trn_loss += self.update_policy(policy_copy, policy_copy_opt, batch_observations, batch_actions, batch_advantages, create_graph=True)

        trn_loss /= self.config["k"]

        # Reset once more so it's a test env
        self.env.set_randomize_env(True)
        self.env.reset()

        # Now test
        observations, clean_actions, noisy_actions, rewards, terminals = self.make_rollout(policy_copy)
        observations_T = T.from_numpy(np.array(observations))
        actions_T = T.from_numpy(np.array(noisy_actions))
        rewards_T = T.from_numpy(np.array(rewards))
        terminals_T = T.from_numpy(np.array(terminals))

        advantages_T = self.calc_advantages_MC(self.config["gamma"], rewards_T, terminals_T)

        log_probs = policy.log_probs(observations_T, actions_T)
        tst_loss = -T.mean(log_probs * advantages_T)
        grad = T.autograd.grad(tst_loss, policy.parameters())

        return grad, trn_loss.detach().numpy(), tst_loss.detach().numpy()

    def meta_train(self):
        meta_trn_opt = T.optim.SGD(self.policy.parameters(),
                                   lr=self.config["learning_rate_meta"],
                                   momentum=self.config["momentum_meta"],
                                   weight_decay=self.config["w_decay_meta"])

        # Perform * iters of meta training
        for mt in range(self.config["n_meta_iters"]):
            meta_trn_opt.zero_grad()
            meta_grads = []
            mean_trn_losses = []
            mean_tst_losses = []

            # Calculate gradient for * env samples
            for _ in range(self.config["batchsize_meta"]):
                # Make n_rollouts on env sample
                meta_grad, mean_trn_loss, mean_tst_loss = self.comp_gradient_on_env_sample(self.policy)
                meta_grads.append(meta_grad)
                mean_trn_losses.append(mean_trn_loss)
                mean_tst_losses.append(mean_tst_loss)

            # Aggregate all meta_gradients
            for meta_grad in meta_grads:
                # Add to meta gradients
                for mg, p in zip(meta_grad, self.policy.parameters()):
                    if p.grad is None:
                        p.grad = mg.clone()
                    else:
                        p.grad += mg.clone()

            # Divide gradient by batchsize
            for p in self.policy.parameters():
                p.grad /= self.config["batchsize_meta"]

            # Update meta parameters
            meta_trn_opt.step()
            print("Meta iter: {}/{}, trn_mean_loss: {}, tst_mean_loss: {}".
                  format(mt, self.config["n_meta_iters"], np.mean(mean_trn_losses), np.mean(mean_tst_losses)))


    def test(self, env, model, deterministic=True):
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


def main():
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

    maml_rl_trainer = MAMLRLTrainer(env, policy, config)

    if config["train"] or socket.gethostname() == "goedel":
        t1 = time.time()
        maml_rl_trainer.meta_train()
        t2 = time.time()

        print("Training time: {}".format(t2 - t1))
        print(config)

    if config["test"] and socket.gethostname() != "goedel":
        if not args["train"]:
            policy.load_state_dict(T.load(config["test_agent_path"]))
        maml_rl_trainer.test(env, policy)

if __name__ == "__main__":
    main()







