import optuna
from baselines3_RL import *
from copy import deepcopy

def objective(trial, config):
    # Hexapod
    config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 2e-4, 7e-4))
    config["gamma"] = trial.suggest_loguniform('gamma', 0.99, 0.999)
    config["phase_scalar"] = trial.suggest_uniform('phase_scalar', 0.05, 0.5)
    #config["phase_decay"] = trial.suggest_loguniform('phase_decay', 0.85, 0.99)
    #config["z_aux_scalar"] = trial.suggest_uniform('z_aux_scalar', 0.07, 0.14)

    # Quad
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
    # algo_config = my_utils.read_config("configs/a2c_quadrotor_config.yaml")
    # env_config = my_utils.read_config("../../envs/bullet_quadrotor/configs/default.yaml")

    algo_config = my_utils.read_config("configs/a2c_hexapod_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_hexapod/configs/joint_phases.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 5000000
    config["verbose"] = False
    config["animate"] = False
    #config["default_session_ID"] = "OPT_HEX"
    config["tensorboard_log"] = False
    config["dummy_vec_env"] = False
    config["N_test"] = 50
    N_trials = 70

    t1 = time.time()
    #study = optuna.create_study(direction='maximize', study_name="hexapod_opt_study", storage='sqlite:///hexapod_opt.db', load_if_exists=True)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda x : objective(x, config), n_trials=N_trials, show_progress_bar=True)
    t2 = time.time()
    print("Time taken: ", t2-t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)





