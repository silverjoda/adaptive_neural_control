import optuna
from baselines_RL import *
from copy import deepcopy
import sqlite3
import sqlalchemy.exc

def objective(trial, config):
    # Hexapod
    config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 7e-4, 3e-3))
    config["gamma"] = trial.suggest_loguniform('gamma', 0.96, 0.99)
    config["ou_sigma"] = trial.suggest_uniform('ou_sigma', 0.5, 1.0)
    config["batchsize"] = trial.suggest_int('batchsize', 32, 512)
    jrl = [-0.6,
           trial.suggest_uniform('jrl_femur', -1.4, -0.6),
           trial.suggest_uniform('jrl_tibia', -0.8, 0.8)]
    jr_diff = [1.2,
               trial.suggest_uniform('jr_diff_femur', 1.5, 2.5),
               trial.suggest_uniform('jr_diff_tibia', 1.5, 2.5)]

    config["joints_rads_low"] = jrl
    config["joints_rads_diff"] = [jrl[i] + jr_diff[i] for i in range(3)]
    config["joints_rads_high"] = [jrl[i] + jr_diff[i] for i in range(3)]

    env, model, _, stats_path = setup_train(config, setup_dirs=True)
    model.learn(total_timesteps=config["iters"])

    eval_env = setup_eval(config, stats_path, seed=1337)
    model.set_env(eval_env)
    avg_episode_rew = test_agent_rnn(eval_env, model, deterministic=True, N=config["N_test"], print_rew=False)
    avg_episode_rew /= config["N_test"]

    try:
        best_value = trial.study.best_value
    except:
        best_value = -1e5
    if avg_episode_rew > best_value:
        model.save("agents/OBSTACLE_A2C_RNN_OPTUNA_policy")
        print("Saved best policy")

    env.close()
    eval_env.close()
    del env
    del eval_env
    del model

    return avg_episode_rew

if __name__ == "__main__":
    algo_config = my_utils.read_config("configs/a2c_hexapod_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_hexapod/configs/wp_obstacle.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 3000000
    config["verbose"] = False
    config["animate"] = False
    config["tensorboard_log"] = False
    config["dummy_vec_env"] = True
    config["policy_name"] = "MlpLstmPolicy"
    config["training_difficulty"] = 0.4
    config["training_difficulty_increment"] = 0.0007  # 0.0005
    config["N_test"] = 70
    N_trials = 70

    t1 = time.time()
    study = optuna.create_study(direction='maximize', study_name="hexapod_obstacle_RNN_study", storage='sqlite:///hexapod_obstacle_rnn.db', load_if_exists=True)

    while True:
        try:
            study.optimize(lambda x : objective(x, config), n_trials=N_trials, show_progress_bar=True)
            break
        except (sqlite3.OperationalError, sqlalchemy.exc.InvalidRequestError):
            print("Optimize failed, restarting")

    t2 = time.time()
    print("Time taken: ", t2-t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)





