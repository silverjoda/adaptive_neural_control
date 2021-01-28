import optuna
from baselines3_RL import *
from copy import deepcopy


def objective(trial, config):
    config["n_steps"] = trial.suggest_int('n_steps', 20, 50)
    config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 1e-4, 8e-4))
    config["gamma"] = trial.suggest_loguniform('gamma', 0.985, 0.999)
    config["policy_hid_dim"] = trial.suggest_int("policy_hid_dim", 64, 256)
    config["phase_scalar"] = trial.suggest_uniform('phase_scalar', 0.03, 0.5)
    config["phase_decay"] = trial.suggest_loguniform('phase_decay', 0.9, 0.99)
    config["z_aux_scalar"] = trial.suggest_uniform('z_aux_scalar', 0.05, 0.1)

    # config["n_steps"] = trial.suggest_int('n_steps', 6, 40)
    # config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 1e-4, 7e-4))
    # config["gamma"] = trial.suggest_loguniform('gamma', 0.985, 0.999)
    # config["policy_hid_dim"] = trial.suggest_int("policy_hid_dim", 48, 256)

    env, model, _, stats_path = setup_train(config, setup_dirs=True)
    model.learn(total_timesteps=config["iters"])
    env.save(stats_path)

    eval_env = setup_eval(config, stats_path, seed=1337)
    model.set_env(eval_env)
    avg_episode_rew = test_agent(eval_env, model, deterministic=True, N=config["N_test"], render=False, print_rew=False)

    env.close()
    eval_env.close()
    del env
    del eval_env
    del model

    return avg_episode_rew[0] / config["N_test"]

if __name__ == "__main__":
    # env_fun = my_utils.import_env("quadrotor_stab")
    # algo_config = my_utils.read_config("configs/a2c_quadrotor_config.yaml")
    # env_config = my_utils.read_config("../../envs/bullet_quadrotor/configs/default.yaml")

    env_fun = my_utils.import_env("hexapod_wp_eef")
    algo_config = my_utils.read_config("configs/a2c_hexapod_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_hexapod/configs/eef.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 30000
    config["verbose"] = True
    config["animate"] = False
    config["default_session_ID"] = "OPT_HEX"
    config["tensorboard_log"] = False
    config["dummy_vec_env"] = True
    config["N_test"] = 10
    N_trials = 1

    # TODO: keep testing single threaded optuna_baselines vs parallel variant to see if it does what it says on the tin.

    t1 = time.time()
    #study = optuna.create_study(direction='maximize', study_name="example-study", storage='sqlite:///example.db', load_if_exists=True)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda x : objective(x, config), n_trials=N_trials, show_progress_bar=True)
    t2 = time.time()
    print("Time taken: ", t2-t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)





