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

class ReplayBuffer:
    def __init__(self):
        self.buffer = {}
    def add(self, k, v):
        self.buffer[k].append(v)
    def get_contents(self):
        obs_dict = {"observations": [],
                    "actions": [],
                    "rewards": [],
                    "terminals": []}
        for k in self.buffer.keys():
            obs_dict[k] = T.tensor(self.buffer[k])
        self.buffer = {}

        return obs_dict
    def __len__(self):
        return len(self.buffer["observations"])


class ACTrainer:
    def __init__(self, config):
        self.config = config

    def make_rollout(self):
        obs = self.env.reset()

        while True:
            self.replay_buffer.add("observations", obs)

            act_np = self.policy.sample_par_action(obs)
            obs, r, done, _ = self.env.step(act_np)

            if config["animate"]:
                self.env.render()

            self.replay_buffer.add("actions", act_np)
            self.replay_buffer.add("rewards", r)

            if done:
                self.replay_buffer.add("terminals", True)
                break
            self.replay_buffer.add("terminals", False)

            # Update
            if len(self.replay_buffer) >= self.config["n_steps"]:
                self.update()

            self.global_step_ctr += 1

    def update(self):
        data = replay_buffer.get_contents()

        if self.config["advantages_MC"]:
            batch_advantages = self.calc_advantages_MC(data["rewards"], data["terminals"])
        else:
            batch_advantages = self.calc_advantages(data["observations"], data["rewards"], data["terminals"])
        loss_policy = self.update_policy(data["observations"], data["actions"], batch_advantages.detach())

        if config["advantages_MC"]:
            loss_vf = 0
        else:
            loss_vf = self.update_vf(batch_advantages)

        # Post update log
        if config["tb_writer"] is not None:
            config["tb_writer"].add_histogram("Batch/Advantages", batch_advantages, global_step=self.global_step_ctr)
            config["tb_writer"].add_scalar("Batch/Loss_policy", loss_policy, global_step=self.global_step_ctr)

            config["tb_writer"].add_histogram("Batch/Rewards", data["rewards"], global_step=self.global_step_ctr)
            config["tb_writer"].add_histogram("Batch/Observations", data["observations"],
                                              global_step=self.global_step_ctr)
            config["tb_writer"].add_histogram("Batch/Sampled actions", data["actions"],
                                              global_step=self.global_step_ctr)
            config["tb_writer"].add_scalar("Batch/Terminal step", len(data["terminals"]) / config["batchsize"],
                                           global_step=self.global_step_ctr)

        print(
            "N_total_steps_train {}/{}, loss_policy: {}, loss_vf: {}, mean ep_rew: {}".
            format(self.global_step_ctr,
                   config["n_total_steps_train"],
                   loss_policy,
                   loss_vf,
                   data["rewards"].sum() / config["batchsize"]))

    def train(self):
        while self.global_step_ctr < self.config["n_total_steps_train"]:
            self.make_rollout()

            # Decay log_std
            # policy.log_std -= config["log_std_decay"]
            # print(policy.log_std)

            if self.global_step_ctr % 1000000 == 0 and self.config["save_policy"]:
                T.save(policy.state_dict(), self.config["sdir"])
                print("Saved checkpoint at {} with params {}".format(self.config["sdir"], self.config))

        if self.config["save_policy"]:
            T.save(self.policy.state_dict(), self.config["sdir"])
            print("Finished training, saved policy at {} with params {}".format(self.config["sdir"], self.config))

    def update_policy(self, batch_states, batch_actions, batch_advantages):
        # Get action log probabilities
        log_probs = self.policy.log_probs(batch_states, batch_actions)

        # Calculate loss function
        loss = -T.mean(log_probs * batch_advantages)

        # Backward pass on policy
        self.policy_optim.zero_grad()
        loss.backward()

        # Step policy update
        self.policy_optim.step()

        return loss.data

    def update_vf(self, batch_advantages):
        self.vf_optim.zero_grad()
        loss = T.mean(0.5 * T.pow(batch_advantages, 2))
        loss.backward()
        self.vf_optim.step()
        return loss.data.detach()

    def calc_advantages_MC(self, batch_rewards, batch_terminals):
        # Monte carlo estimate of targets
        targets = []
        with T.no_grad():
            for r, t in zip(reversed(batch_rewards), reversed(batch_terminals)):
                if t:
                    R = r
                else:
                    R = r + self.config["gamma"] * R
                targets.append(R.view(1, 1))
            targets = T.cat(list(reversed(targets)))
        return targets

    def calc_advantages(self, batch_observations, batch_rewards, batch_terminals):
        batch_values = vf(batch_observations)
        targets = []
        for i in reversed(range(len(batch_rewards))):
            r, v, t = batch_rewards[i], batch_values[i], batch_terminals[i]
            if t:
                R = r
            else:
                v_next = batch_values[i + 1]
                R = r + self.config["gamma"] * v_next - v
            targets.append(R.view(1, 1))
        targets = T.cat(list(reversed(targets)))
        return targets

    def test_agent(self, N=100, print_rew=False, render=True):
        total_rew = 0
        for i in range(N):
            obs = self.env.reset()
            episode_rew = 0
            while True:
                action = self.policy.sample_action(obs)
                obs, reward, done, info = self.env.step(action)
                episode_rew += reward
                total_rew += reward
                if render:
                    self.env.render()
                if done:
                    if print_rew:
                        print(episode_rew)
                    break
        return total_rew

    def setup(self, setup_dirs=True):
        if setup_dirs:
            for s in ["agents", "agents_cp", "tb"]:
                if not os.path.exists(s):
                    os.makedirs(s)

        # Random ID of this session
        if self.config["default_session_ID"] is None:
            self.config["session_ID"] = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))
        else:
            self.config["session_ID"] = "TST"

        # Import correct env by name
        self.env_fun = my_utils.import_env(self.config["env_name"])
        #env = env_fun(config)
        self.env = VecNormalize(SubprocVecEnv([lambda: self.env_fun(self.config) for _ in range(self.config["n_envs"])], start_method='fork'),
                           gamma=self.config["gamma"],
                           norm_obs=self.config["norm_obs"],
                           norm_reward=self.config["norm_reward"])

        self.policy = my_utils.make_par_policy(env, self.config)
        self.vf = my_utils.make_par_vf(env, self.config)

        self.config["tb_writer"] = None
        if self.config["log_tb"] and setup_dirs:
            tb_writer = SummaryWriter(f'tb/{self.config["session_ID"]}')
            config["tb_writer"] = tb_writer

        self.config["sdir"] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            f'agents/{self.config["session_ID"]}_AC_policy.p')

        self.policy_optim = None
        self.vf_optim = None
        if self.config["policy_optim"] == "rmsprop":
            self.policy_optim = T.optim.RMSprop(policy.parameters(),
                                           lr=self.config["policy_learning_rate"],
                                           weight_decay=self.config["weight_decay"],
                                           eps=1e-8, momentum=self.config["momentum"])
            self.vf_optim = T.optim.RMSprop(vf.parameters(),
                                       lr=self.config["vf_learning_rate"],
                                       weight_decay=self.config["weight_decay"],
                                       eps=1e-8, momentum=self.config["momentum"])
        if self.config["policy_optim"] == "sgd":
            self.policy_optim = T.optim.SGD(policy.parameters(),
                                       lr=self.config["policy_learning_rate"],
                                       weight_decay=self.config["weight_decay"],
                                       momentum=self.config["momentum"])
            self.vf_optim = T.optim.SGD(vf.parameters(),
                                   lr=self.config["vf_learning_rate"],
                                   weight_decay=self.config["weight_decay"],
                                   momentum=self.config["momentum"])
        if self.config["policy_optim"] == "adam":
            self.policy_optim = T.optim.Adam(policy.parameters(),
                                        lr=self.config["policy_learning_rate"],
                                        weight_decay=self.config["weight_decay"])
            self.vf_optim = T.optim.Adam(vf.parameters(),
                                    lr=self.config["vf_learning_rate"],
                                    weight_decay=self.config["weight_decay"])
        assert self.policy_optim is not None

        self.replay_buffer = ReplayBuffer()

        self.global_step_ctr = 0

        return self.env, self.policy, self.vf, self.policy_optim, self.vf_optim, self.replay_buffer

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

if __name__=="__main__":
    args = parse_args()
    algo_config = my_utils.read_config(args["algo_config"])
    env_config = my_utils.read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    print(config)

    ac_trainer = ACTrainer(config)
    env, policy, vf, policy_optim, vf_optim, replay_buffer = ac_trainer.setup()

    if config["train"] or socket.gethostname() == "goedel":
        t1 = time.time()
        ac_trainer.train()
        t2 = time.time()

        print("Training time: {}".format(t2 - t1))
        print(config)

    if config["test"] and socket.gethostname() != "goedel":
        if not args["train"]:
            policy.load_state_dict(T.load(config["test_agent_path"]))
        ac_trainer.test_agent(print_rew=True)


