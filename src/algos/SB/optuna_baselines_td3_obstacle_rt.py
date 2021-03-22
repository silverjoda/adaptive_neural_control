import optuna
from baselines3_TD3 import *
import sqlite3
import sqlalchemy.exc
from copy import deepcopy

def objective(trial, config):
    # Hexapod
    config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 3e-4, 3e-3))
    config["gamma"] = trial.suggest_loguniform('gamma', 0.96, 0.98)
    config["ou_sigma"] = trial.suggest_uniform('ou_sigma', 0.2, 0.5)
    config["ou_theta"] = trial.suggest_uniform('ou_theta', 0.03, 0.3)
    config["ou_dt"] = trial.suggest_uniform('ou_dt', 0.03, 0.3)
    config["batchsize"] = trial.suggest_int('batchsize', 96, 190)
    config["max_steps"] = trial.suggest_int('max_steps', 60, 300)
    config["training_difficulty"] = trial.suggest_uniform('training_difficulty', 0.7, 0.95)
    config["training_difficulty_increment"] = trial.suggest_uniform('training_difficulty_increment', 0.00005, 0.0005)

    env, _, _, stats_path = setup_train(config, setup_dirs=True)
    model = TD3.load("agents/OBSTACLE_TD3_OPTUNA_policy")
    model.set_env(env)
    model.learn(total_timesteps=config["iters"])

    config["training_difficulty"] = 1.0
    config["max_steps"] = 100
    eval_env = setup_eval(config, stats_path, seed=1337)
    model.set_env(eval_env)
    avg_episode_rew = test_agent(eval_env, model, deterministic=True, N=config["N_test"], render=False, print_rew=False)
    avg_episode_rew /= config["N_test"]

    try:
        best_value = trial.study.best_value
    except:
        best_value = -1e5
    if avg_episode_rew > best_value:
        model.save("agents/OBSTACLE_RT_TD3_OPTUNA_policy")
        print("Saved best policy")

    env.close()
    eval_env.close()
    del env
    del eval_env
    del model

    return avg_episode_rew

if __name__ == "__main__":
    algo_config = my_utils.read_config("configs/td3_default_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_hexapod/configs/wp_obstacle.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 1000000
    config["verbose"] = False
    config["animate"] = False
    #config["default_session_ID"] = "OPT_HEX"
    config["tensorboard_log"] = False
    config["dummy_vec_env"] = False
    config["N_test"] = 50
    N_trials = 100

    t1 = time.time()
    study = optuna.create_study(direction='maximize', study_name="hexapod_obstacle_rt_study", storage='sqlite:///hexapod_obstacle_rt.db', load_if_exists=True)

    while True:
        try:
            study.optimize(lambda x : objective(x, config), n_trials=N_trials, show_progress_bar=True)
            break
        except (sqlite3.OperationalError, sqlalchemy.exc.InvalidRequestError):
            print("Optimize failed, restarting")

    t2 = time.time()
    print("Time taken: ", t2-t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)





