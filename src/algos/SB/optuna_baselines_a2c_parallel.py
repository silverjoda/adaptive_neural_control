import optuna
from baselines3_RL import *
from copy import deepcopy
from multiprocessing import Pool, Process, Manager

def train_and_eval(config, procnum, return_dict):
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

    return_dict[procnum] = avg_episode_rew[0] / config["N_test"]

def objective(trial, config):
    # Hexapod
    config["n_steps"] = trial.suggest_int('n_steps', 10, 35)
    config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 2e-4, 7e-4))
    config["gamma"] = trial.suggest_loguniform('gamma', 0.993, 0.999)
    config["policy_hid_dim"] = trial.suggest_int("policy_hid_dim", 96, 196)
    config["phase_scalar"] = trial.suggest_uniform('phase_scalar', 0.1, 0.3)
    config["phase_decay"] = trial.suggest_loguniform('phase_decay', 0.85, 0.99)
    config["z_aux_scalar"] = trial.suggest_uniform('z_aux_scalar', 0.07, 0.14)

    # Quad
    # config["n_steps"] = trial.suggest_int('n_steps', 6, 40)
    # config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 1e-4, 7e-4))
    # config["gamma"] = trial.suggest_loguniform('gamma', 0.985, 0.999)
    # config["policy_hid_dim"] = trial.suggest_int("policy_hid_dim", 48, 256)

    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(config["N_reps"]):
        p = Process(target=train_and_eval, args=(config, i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    return np.mean([v for v in return_dict.values()])

if __name__ == "__main__":
    # env_fun = my_utils.import_env("quadrotor_stab")
    # algo_config = my_utils.read_config("configs/a2c_quadrotor_config.yaml")
    # env_config = my_utils.read_config("../../envs/bullet_quadrotor/configs/default.yaml")

    time.sleep(np.random.rand())

    env_fun = my_utils.import_env("hexapod_wp_eef")
    algo_config = my_utils.read_config("configs/a2c_hexapod_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_hexapod/configs/eef.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 16000000
    config["verbose"] = False
    config["animate"] = False
    #config["default_session_ID"] = "OPT_HEX"
    config["tensorboard_log"] = False
    config["dummy_vec_env"] = True
    config["N_test"] = 50
    config["N_reps"] = 5
    N_trials = 1

    t1 = time.time()
    study = optuna.create_study(direction='maximize', study_name="hexapod_opt_study", storage='sqlite:///hexapod_opt.db', load_if_exists=True)
    #study = optuna.create_study(direction='maximize')
    study.optimize(lambda x : objective(x, config), n_trials=N_trials, show_progress_bar=True)
    t2 = time.time()
    print("Time taken: ", t2 - t1)

    print("Best params: ", study.best_params, " Best value: ", study.best_value)





