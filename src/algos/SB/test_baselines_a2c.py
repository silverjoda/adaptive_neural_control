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
#from src.envs.bullet_cartpole.cartpole.cartpole import CartPoleBulletEnv as env_dcp
#from src.envs.bullet_cartpole.hangpole_goal.hangpole_goal import HangPoleGoalBulletEnv as env_dcp
#from src.envs.bullet_cartpole.double_cartpole_goal.double_cartpole_goal import DoubleCartPoleBulletEnv as env_dcp
#from src.envs.bullet_cartpole.cartpole_swingup.cartpole_swingup import CartPoleSwingUpBulletEnv as env_fun
#from src.envs.bullet_cartpole.double_cartpole_swingup_goal_variable.double_cartpole_swingup_goal_variable import DoubleCartpoleSwingupGoalVariable as env_fun
#from src.envs.bullet_cartpole.hangpole_goal_cont_variable.hangpole_goal_cont_variable import HangPoleGoalContVariableBulletEnv as env_fun
from src.envs.bullet_nexabot.quadruped.quadruped import QuadrupedBulletEnv as env_fun

def make_env():
    def _init():
        env = env_fun(animate=False, max_steps=300)
        return env
    return _init

if __name__ == "__main__":
    TRAIN = True

    if TRAIN:
        env = SubprocVecEnv([make_env() for _ in range(1)])
        model = A2C('MlpPolicy', env, learning_rate=1e-3, verbose=1, n_steps=32, tensorboard_log="/tmp", gamma=0.99)
        # Train the agent
        t1 = time.time()
        model.learn(total_timesteps=int(30000))
        t2 = time.time()
        print("Training time: {}".format(t2-t1))
        model.save("a2c_mdl")
        del model
        env.close()

    env = env_fun(animate=True, max_steps=300)
    # Load the trained agent
    model = A2C.load("a2c_mdl")
    print(evaluate_policy(model, env, n_eval_episodes=3))

    obs = env.reset()
    for _ in range(100):
        cum_rew = 0
        for i in range(800):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            env.render()
            if done:
                obs = env.reset()
                print(cum_rew)
                break

    env.close()