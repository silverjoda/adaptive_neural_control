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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

class ReplayBuffer:
    def __init__(self):
        self.buffer = self._init_buffer()

    def _init_buffer(self):
        return {"observations": [],
                "actions": [],
                "rewards": [],
                "terminals": []}

    def add(self, k, v):
        self.buffer[k].append(T.tensor(v))

    def get_contents_and_clear(self):
        obs_dict = self._init_buffer()
        for k in self.buffer.keys():
            obs_dict[k] = T.stack(self.buffer[k])
        self.buffer = self._init_buffer()
        return obs_dict

    def __len__(self):
        return len(self.buffer["observations"])


class ACTrainer:
    def __init__(self, config):
        self.config = config

    def update(self):
        data = self.replay_buffer.get_contents_and_clear()

        batch_values, batch_advantages = self.calc_advantages(data["observations"], data["rewards"], data["terminals"])

        batch_states_flat = data["observations"].view(-1, data["observations"].shape[-1])
        batch_actions_flat = data["actions"].view(-1, data["actions"].shape[-1])
        batch_advantages_flat = batch_advantages.view(-1)

        # Get action log probabilities
        log_probs = self.policy.log_probs(batch_states_flat, batch_actions_flat)

        # Calculate loss function
        loss_policy = -T.mean(log_probs * batch_advantages_flat.detach())
        loss_vf = T.mean(0.5 * T.pow(batch_advantages, 2))

        loss = loss_policy + loss_vf

        self.policy_optim.zero_grad()
        self.vf_optim.zero_grad()
        loss.backward()

        self.vf_optim.step()
        self.policy_optim.step()

        self.n_updates += 1

        # Post update log
        if config["tb_writer"] is not None:
            config["tb_writer"].add_histogram("Batch/Advantages", batch_advantages, global_step=self.global_step_ctr)
            config["tb_writer"].add_histogram("Batch/Values", batch_values, global_step=self.global_step_ctr)
            config["tb_writer"].add_histogram("Batch/Logprobs", log_probs, global_step=self.global_step_ctr)
            config["tb_writer"].add_scalar("Batch/Loss_policy", self.loss_policy, global_step=self.global_step_ctr)
            config["tb_writer"].add_scalar("Batch/Loss_vf", self.loss_vf, global_step=self.global_step_ctr)

            config["tb_writer"].add_histogram("Batch/Rewards", data["rewards"], global_step=self.global_step_ctr)
            config["tb_writer"].add_histogram("Batch/Observations", data["observations"],
                                              global_step=self.global_step_ctr)
            config["tb_writer"].add_histogram("Batch/Sampled actions", data["actions"],
                                              global_step=self.global_step_ctr)
            config["tb_writer"].add_scalar("Batch/Terminal step", len(data["terminals"]) / self.config["batchsize"],
                                           global_step=self.global_step_ctr)

        if self.n_updates % 30 == 0:
            mean_eval_rews = self.eval_agent_par(N=3)
            print("N_total_steps_train {}/{}, loss_policy: {}, loss_vf: {}, mean ep_rew: {}".
                  format(self.global_step_ctr,
                         self.config["n_total_steps_train"],
                         loss_policy,
                         loss_vf,
                         mean_eval_rews))


    def train(self):
        t1 = time.time()

        obs = self.env.reset()
        while self.global_step_ctr < self.config["n_total_steps_train"]:
            self.replay_buffer.add("observations", obs)

            act_np = self.policy.sample_par_action(obs)
            obs, r, done, _ = self.env.step(act_np)

            if config["animate"]:
                self.env.render()

            self.replay_buffer.add("actions", act_np)
            self.replay_buffer.add("rewards", r)
            self.replay_buffer.add("terminals", done)

            # Update
            if len(self.replay_buffer) >= self.config["n_steps_per_update"]:
                self.update()

            self.global_step_ctr += self.config["n_envs"]

            # Decay log_std
            # policy.log_std -= config["log_std_decay"]
            # print(policy.log_std)

            if self.global_step_ctr % self.config["n_steps_per_checkpoint"] == 0 and self.config["save_policy"]:
                T.save(self.policy.state_dict(), self.config["sdir"])
                print("Saved checkpoint at {} with params {}".format(self.config["sdir"], self.config))

        if self.config["save_policy"]:
            T.save(self.policy.state_dict(), self.config["sdir"])
            print("Finished training, saved policy at {} with params {}".format(self.config["sdir"], self.config))
        t2 = time.time()
        print("Training time: {}".format(t2 - t1))


    def calc_advantages(self, batch_observations, batch_rewards, batch_terminals):
        batch_values = self.vf(batch_observations).squeeze(2)
        targets = []
        for i in reversed(range(len(batch_rewards))):
            r, v, t = batch_rewards[i], batch_values[i], batch_terminals[i]
            if i == len(batch_rewards) - 1:
                A = r - v
            else:
                A = r - v + self.config["gamma"] * batch_values[i + 1] * T.logical_not(t)
            targets.append(A.view(1, self.config["n_envs"]))
        targets = T.cat(list(reversed(targets)))
        return batch_values, targets

    def eval_agent_par(self, N=10):
        total_rew = 0
        for i in range(N):
            obs = self.env.reset()
            done_arr = np.array([False] * self.config["n_envs"])
            while True:
                action = self.policy.sample_action(obs)
                obs, reward, done, info = self.env.step(action)
                total_rew += np.mean(reward * np.logical_not(done_arr))
                done_arr = np.logical_or(done_arr, done)
                if done_arr.all():
                    break
        return total_rew / N

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

                self.env.render()

                if done:
                    if print_rew:
                        print(episode_rew)
                    break
        return total_rew

    def setup_train(self, setup_dirs=True):
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
        self.env = VecNormalize(SubprocVecEnv([lambda: self.env_fun(self.config) for _ in range(self.config["n_envs"])], start_method='fork'),
                           gamma=self.config["gamma"],
                           norm_obs=self.config["norm_obs"],
                           norm_reward=self.config["norm_reward"])

        self.policy = my_utils.make_par_policy(self.env, self.config)
        self.vf = my_utils.make_par_vf(self.env, self.config)

        self.config["tb_writer"] = None
        if self.config["log_tb"] and setup_dirs:
            tb_writer = SummaryWriter(f'tb/{self.config["session_ID"]}')
            config["tb_writer"] = tb_writer

        self.config["sdir"] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            f'agents/{self.config["session_ID"]}_AC_policy.p')

        self.policy_optim = None
        self.vf_optim = None
        if self.config["policy_optim"] == "rmsprop":
            self.policy_optim = T.optim.RMSprop(self.policy.parameters(),
                                           lr=self.config["policy_learning_rate"],
                                           weight_decay=self.config["weight_decay"],
                                           eps=1e-8, momentum=self.config["momentum"])
            self.vf_optim = T.optim.RMSprop(self.vf.parameters(),
                                       lr=self.config["vf_learning_rate"],
                                       weight_decay=self.config["weight_decay"],
                                       eps=1e-8, momentum=self.config["momentum"])
        if self.config["policy_optim"] == "sgd":
            self.policy_optim = T.optim.SGD(self.policy.parameters(),
                                       lr=self.config["policy_learning_rate"],
                                       weight_decay=self.config["weight_decay"],
                                       momentum=self.config["momentum"])
            self.vf_optim = T.optim.SGD(self.vf.parameters(),
                                   lr=self.config["vf_learning_rate"],
                                   weight_decay=self.config["weight_decay"],
                                   momentum=self.config["momentum"])
        if self.config["policy_optim"] == "adam":
            self.policy_optim = T.optim.Adam(self.policy.parameters(),
                                        lr=self.config["policy_learning_rate"],
                                        weight_decay=self.config["weight_decay"])
            self.vf_optim = T.optim.Adam(self.vf.parameters(),
                                    lr=self.config["vf_learning_rate"],
                                    weight_decay=self.config["weight_decay"])
        assert self.policy_optim is not None

        self.replay_buffer = ReplayBuffer()

        self.global_step_ctr = 0
        self.n_updates = 0
        self.loss_policy = 0
        self.loss_vf = 0

        return self.env, self.policy, self.vf, self.policy_optim, self.vf_optim, self.replay_buffer

    def setup_test(self):
        env_fun = my_utils.import_env(env_config["env_name"])
        self.env = DummyVecEnv([lambda: env_fun(config)])
        self.policy = my_utils.make_par_policy(self.env, config)
        self.policy.load_state_dict(T.load(config["test_agent_path"]))


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
    parser.add_argument('--env_config', type=str, default="default.yaml", required=False,
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

    if config["train"] or socket.gethostname() == "goedel":
        ac_trainer.setup_train()
        ac_trainer.train()

        print(config)

    if config["test"] and socket.gethostname() != "goedel":
        ac_trainer.setup_test()
        ac_trainer.test_agent(print_rew=True)


