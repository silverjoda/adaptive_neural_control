import optuna
from baselines3_TD3 import *
import sqlite3
import sqlalchemy.exc
import pickle

def objective(trial, config):
    # Hexapod
    config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 9e-4, 3e-3))
    config["gamma"] = trial.suggest_loguniform  ('gamma', 0.95, 0.98)
    config["ou_sigma"] = trial.suggest_uniform('ou_sigma', 0.3, 0.6)
    config["ou_theta"] = trial.suggest_uniform('ou_theta', 0.03, 0.3)
    config["ou_dt"] = trial.suggest_uniform('ou_dt', 0.03, 0.3)
    config["training_difficulty"] = trial.suggest_uniform('training_difficulty', 0.3, 0.6)
    config["training_difficulty_increment"] = trial.suggest_uniform('training_difficulty_increment', 0.00005, 0.0005)

    config["min_dif_state_queue_len"] = trial.suggest_int('min_dif_state_queue_len', 50, 500)
    config["dif_state_sample_prob"] = trial.suggest_uniform('dif_state_sample_prob', 0.1, 0.7)
    config["low_progress_thresh"] = trial.suggest_uniform('low_progress_thresh', 0.03, 0.2)
    config["progress_pen"] = trial.suggest_uniform('progress_pen', 0.03, 0.15)

    # jrl = [-0.6,
    #        trial.suggest_uniform('jrl_femur', -1.2, -0.8),
    #        trial.suggest_uniform('jrl_tibia', 0.0, 0.5)]
    # jr_diff = [1.2,
    #        trial.suggest_uniform('jr_diff_femur', 1.4, 2.2),
    #        trial.suggest_uniform('jr_diff_tibia', 1.0, 2.2)]

    # config["joints_rads_low"] = jrl
    # config["joints_rads_diff"] = jr_diff

    env, model, _, stats_path = setup_train(config, setup_dirs=True)
    model.learn(total_timesteps=config["iters"])

    config["training_difficulty"] = 1.0
    eval_env = setup_eval(config, stats_path, seed=1337)
    model.set_env(eval_env)
    avg_episode_rew = test_agent(eval_env, model, deterministic=True, N=config["N_test"], render=False, print_rew=False)
    avg_episode_rew /= config["N_test"]

    try:
        best_value = trial.study.best_value
    except:
        best_value = -1e5
    if avg_episode_rew > best_value:
        model.save("agents/OBSTACLE_TD3_OPTUNA_policy")
        print("Saved best policy")

        pickle.dump(config, open( "agents/obstacle_best_params.p", "wb" ) )

    env.close()
    eval_env.close()
    del env
    del eval_env
    del model

    return avg_episode_rew

if __name__ == "__main__":
    algo_config = my_utils.read_config("configs/td3_default_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_hexapod/configs/wp_obstacle.yaml")

    #favorite_color = pickle.load(open("agents/best_params.p", "rb"))

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
    study = optuna.create_study(direction='maximize', study_name="hexapod_obstacle_study", storage='sqlite:///hexapod_obstacle.db', load_if_exists=True)

    while True:
        try:
            study.optimize(lambda x : objective(x, config), n_trials=N_trials, show_progress_bar=True)
            break
        except (sqlite3.OperationalError, sqlalchemy.exc.InvalidRequestError):
            print("Optimize failed, restarting")

    t2 = time.time()
    print("Time taken: ", t2-t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)





