import optuna
from baselines3_TD3 import *
import sqlite3
import sqlalchemy.exc
import pickle

def objective(trial, config):
    config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 3e-4, 5e-3))
    config["gamma"] = trial.suggest_loguniform  ('gamma', 0.95, 0.99)
    config["ou_sigma"] = trial.suggest_uniform('ou_sigma', 0.2, 0.8)
    config["ou_theta"] = trial.suggest_uniform('ou_theta', 0.01, 0.3)
    config["ou_dt"] = trial.suggest_uniform('ou_dt', 0.01, 2.0)

    env, model, _, stats_path = setup_train(config, setup_dirs=True)
    model.learn(total_timesteps=config["iters"])

    eval_env = setup_eval(config, stats_path, seed=1337)
    model.set_env(eval_env)
    avg_episode_rew = test_agent(eval_env, model, deterministic=True, N=config["N_test"], render=False, print_rew=False)
    avg_episode_rew /= config["N_test"]

    try:
        best_value = trial.study.best_value
    except:
        best_value = -1e5
    if avg_episode_rew > best_value:
        model.save("agents/QUAD_TD3_OPTUNA_policy")
        print("Saved best policy")

        pickle.dump(config, open( "agents/quad_best_params.p", "wb" ) )

    env.close()
    eval_env.close()
    del env
    del eval_env
    del model

    return avg_episode_rew

if __name__ == "__main__":
    algo_config = my_utils.read_config("configs/td3_quadrotor_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_quadrotor/configs/default.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 500000
    config["verbose"] = False
    config["animate"] = False
    config["tensorboard_log"] = False
    config["dummy_vec_env"] = False
    config["N_test"] = 50
    N_trials = 100

    t1 = time.time()
    study = optuna.create_study(direction='maximize', study_name="quad_opt_study",
                                storage='sqlite:///quad_opt.db', load_if_exists=True)

    while True:
        try:
            study.optimize(lambda x: objective(x, config), n_trials=N_trials, show_progress_bar=True)
            break
        except (sqlite3.OperationalError, sqlalchemy.exc.InvalidRequestError):
            print("Optimize failed, restarting")

    t2 = time.time()
    print("Time taken: ", t2 - t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)




