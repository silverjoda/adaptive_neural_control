import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
import time
import socket

if __name__ == "__main__":
    # from src.envs.bullet_cartpole.cartpole.cartpole import CartPoleBulletEnv as env_fun
    from src.envs.bullet_cartpole.hangpole_goal.hangpole_goal import HangPoleGoalBulletEnv as env_fun
    # from src.envs.bullet_cartpole.double_cartpole_goal.double_cartpole_goal import DoubleCartPoleBulletEnv as env_fun
    # from src.envs.bullet_cartpole.cartpole_swingup.cartpole_swingup import CartPoleSwingUpBulletEnv as env_fun
    # from src.envs.bullet_cartpole.double_cartpole_swingup_goal_variable.double_cartpole_swingup_goal_variable import DoubleCartpoleSwingupGoalVariable as env_fun
    #from src.envs.bullet_cartpole.hangpole_goal_cont_variable.hangpole_goal_cont_variable import HangPoleGoalContVariableBulletEnv as env_fun
    #from src.envs.bullet_nexabot.quadruped.quadruped import QuadrupedBulletEnv as env_fun

    def make_env():
        def _init():
            env = env_fun(animate=False, max_steps=200)
            return env

        return _init

    def objective(args):
        n_steps = args['n_steps']
        num_hidden = args['num_hidden']
        learning_rate = args['learning_rate']
        gamma = args['gamma']

        env = make_env()()#SubprocVecEnv([make_env() for _ in range(6)])
        policy_kwargs = dict(net_arch=[int(num_hidden), int(num_hidden)])
        model = A2C('MlpPolicy', env, learning_rate=learning_rate, verbose=1, n_steps=int(n_steps), tensorboard_log="/tmp", gamma=gamma, policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=10000)
        env.close()
        env = make_env()()
        mean_rew = evaluate_policy(model, env, n_eval_episodes=3)[0]
        del model
        env.close()
        del env
        return mean_rew

    # define a search space
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

    hyperparameters = {
        'n_steps': hp.quniform('batch_size', 10, 100, 10),
        'num_hidden': hp.quniform('num_hidden', 12, 64, 8),
        'learning_rate': hp.choice('learning_rate', [1e-4, 5e-4, 1e-3, 3e-3, 8e-3]),
        'gamma': hp.choice('gamma', [0.95, 0.97, 0.99, 0.997]),
    }

    # minimize the objective over the space
    from hyperopt import fmin, tpe, space_eval

    trials = Trials()
    best = fmin(objective, hyperparameters, algo=tpe.suggest, trials=trials, max_evals=30)

    print(best)
    print(space_eval(hyperparameters, best))




