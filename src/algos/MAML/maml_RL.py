import argparse
import random
import string
from copy import deepcopy

import numpy as np
import torch as T
import yaml

import src.my_utils as my_utils
import src.policies as policies


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

def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def import_env(name):
    env_fun = None
    if name == "hexapod":
        from src.envs.bullet_nexabot.hexapod.hexapod import HexapodBulletEnv as env_fun
    if name == "quadruped":
        from src.envs.bullet_nexabot.quadruped.quadruped import QuadrupedBulletEnv as env_fun
    if name == "quadrotor":
        from src.envs.bullet_quadrotor.quadrotor import QuadrotorBulletEnv as env_fun
    if name == "buggy":
        from src.envs.bullet_buggy.buggy import BuggyBulletEnv as env_fun
    assert env_fun is not None, "Env name not found, exiting. "
    return env_fun

def make_env(config, env_fun):
    def _init():
        env = env_fun(config)
        return env
    return _init

def make_policy(env, config):
    if config["policy_type"] == "slp":
        return policies.SLP_PG(env, config)
    elif config["policy_type"] == "mlp":
        return policies.NN_PG(env, config)
    elif config["policy_type"] == "rnn":
        return policies.RNN_PG(env, config)
    else:
        raise TypeError

def make_action_noise_policy(env, config):
    return None

def test_agent(env, model, deterministic=True):
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

class MAMLRLTrainer:
    def __init__(self, env, policy, noise_policy, config):
        self.env = env
        self.policy = policy
        self.noise_policy = noise_policy
        self.config = config

    def make_rollout(self, env, policy):
        self.env.set_randomize_env(False)
        obs = self.env.reset()
        observations = []
        next_observations = []
        actions = []
        rewards = []
        while True:
            observations.append(obs)
            act = policy.sample_action(my_utils.to_tensor(obs, True)).squeeze(0).detach().numpy()
            obs, r, done, _ = env.step(act)
            actions.append(act)
            rewards.append(r)
            next_observations.append(obs)
            if done: break
        terminals = [False] * len(observations)
        terminals[-1] = True
        return observations, next_observations, actions, rewards, terminals

    def update_policy_ppo(self, policy, policy_optim, batch_states, batch_actions, batch_advantages,
                          update_iters):
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
            batch_next_observations = []
            batch_terminals = []

            for i in range(self.config["batchsize"]):
                observations, next_observations, actions, rewards, terminals = self.make_rollout(self.env, policy_copy)

                batch_observations.extend(observations)
                batch_next_observations.extend(next_observations)
                batch_actions.extend(actions)
                batch_rewards.extend(rewards)
                batch_terminals.extend(terminals)

            batch_observations = T.from_numpy(np.array(batch_observations))
            #batch_next_observations = T.from_numpy(np.array(batch_next_observations))
            batch_actions = T.from_numpy(np.array(batch_actions))
            batch_rewards = T.from_numpy(np.array(batch_rewards))
            batch_terminals = T.from_numpy(np.array(batch_terminals))

            batch_advantages = self.calc_advantages_MC(self.config["gamma"], batch_rewards, batch_terminals)
            trn_loss+= self.update_policy(policy_copy, policy_copy_opt, batch_observations, batch_actions, batch_advantages, create_graph=True)

        trn_loss /= self.config["k"]

        # Reset once more so it's a test env
        self.env.set_randomize_env(True)
        self.env.reset()

        # Now test
        observations, next_observations, actions, rewards, terminals = self.make_rollout(self.env, policy_copy)
        observations_T = T.from_numpy(np.array(observations))
        # batch_next_observations = T.from_numpy(np.array(batch_next_observations))
        actions_T = T.from_numpy(np.array(actions))
        rewards_T = T.from_numpy(np.array(rewards))
        terminals_T = T.from_numpy(np.array(terminals))

        advantages_T = self.calc_advantages_MC(self.config["gamma"], rewards_T, terminals_T)

        log_probs = policy.log_probs(observations_T, actions_T)
        test_loss = -T.mean(log_probs * advantages_T)
        grad = T.autograd.grad(test_loss, policy.parameters())

        return grad, trn_loss, test_loss

    def meta_train(self, n_meta_iters=10000):
        meta_trn_opt = T.optim.SGD(policy.parameters(),
                                   lr=self.config["learning_rate_meta"],
                                   momentum=self.config["momentum_meta"],
                                   weight_decay=self.config["w_decay_meta"])

        # Perform * iters of meta training
        for mt in range(n_meta_iters):
            meta_trn_opt.zero_grad()
            meta_grads = []
            mean_trn_losses = []
            mean_tst_losses = []

            # Calculate gradient for * env samples
            for _ in range(self.config["batchsize_meta"]):
                # Make n_rollouts on env sample
                meta_grad, mean_trn_loss, mean_tst_loss = self.comp_gradient_on_env_sample(policy)
                meta_grads.append(meta_grad)
                mean_trn_losses.append(mean_trn_loss)
                mean_tst_losses.append(mean_tst_loss)

            # Aggregate all meta_gradients
            for meta_grad in meta_grads:
                # Add to meta gradients
                for mg, p in zip(meta_grad, policy.parameters()):
                    if p.grad is None:
                        p.grad = mg.clone()
                    else:
                        p.grad += mg.clone()

            # Divide gradient by batchsize
            for p in policy.parameters():
                p.grad /= self.config["batchsize_meta"]

            # Update meta parameters
            meta_trn_opt.step()


            print("Meta iter: {}/{}, trn_mean_loss: {}, tst_mean_loss: {}".
                  format(mt, n_meta_iters, np.mean(mean_trn_losses), np.mean(mean_tst_losses)))

        return policy

    def test(self):
        pass

if __name__ == "__main__":
    args = parse_args()
    algo_config = read_config(args["algo_config"])
    env_config = read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    print(config)

    env_fun = import_env(config["env_name"])
    env = env_fun(config)
    noise_policy = make_action_noise_policy(env, config)
    policy = make_policy(env, config)

    session_ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

    maml_rl_trainer = MAMLRLTrainer(env, policy, noise_policy, config)
    maml_rl_trainer.meta_train(n_meta_iters=args["n_meta_iters"])
    maml_rl_trainer.test()



