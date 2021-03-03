import optuna
from baselines3_TD3 import *
import sqlite3
import sqlalchemy.exc

def objective(trial, config):
    # Hexapod
    config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 1e-3, 3e-3))
    config["gamma"] = trial.suggest_loguniform('gamma', 0.95, 0.97)
    config["ou_sigma"] = trial.suggest_uniform('ou_sigma', 0.30, 0.50)
    config["batchsize"] = trial.suggest_int('batchsize', 32, 196)

    jrl = [-0.5,
           trial.suggest_uniform('jrl_femur', -1.0, -1.01),
           trial.suggest_uniform('jrl_tibia', 0.5, 0.51)]
    jr_diff = [1.0,
               trial.suggest_uniform('jr_diff_femur', 0.99, 1.01),
               trial.suggest_uniform('jr_diff_tibia', 0.99, 1.01)]

    config["joints_rads_low"] = jrl
    config["joints_rads_diff"] = [jrl[i] + jr_diff[i] for i in range(3)]
    config["joints_rads_high"] = [jrl[i] + jr_diff[i] for i in range(3)]

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
        model.save(f"agents/{config['training_mode']}_TD3_OPTUNA_policy")
        print("Saved best policy")

    env.close()
    eval_env.close()
    del env
    del eval_env
    del model

    return avg_episode_rew

if __name__ == "__main__":
    algo_config = my_utils.read_config("configs/td3_default_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_hexapod/configs/hexapod.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 500000
    config["verbose"] = False
    config["animate"] = False
    #config["default_session_ID"] = "OPT_HEX"
    config["tensorboard_log"] = False
    config["dummy_vec_env"] = False
    config["N_test"] = 50
    N_trials = 100

    t1 = time.time()
    study = optuna.create_study(direction='maximize', study_name="hexapod_rugged_opt_study", storage='sqlite:///hexapod_rugged_opt.db', load_if_exists=True)
    #study = optuna.create_study(direction='maximize')

    while True:
        try:
            study.optimize(lambda x : objective(x, config), n_trials=N_trials, show_progress_bar=True)
            break
        except (sqlite3.OperationalError, sqlalchemy.exc.InvalidRequestError):
            print("Optimize failed, restarting")

    t2 = time.time()
    print("Time taken: ", t2-t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)





