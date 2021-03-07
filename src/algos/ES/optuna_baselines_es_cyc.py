import optuna
import src.my_utils as my_utils
import time
import os
import socket
import torch as T
import random
from es import *

def objective(trial, config):
    # Hexapod
    config["std"] = trial.suggest_uniform('std', 0.1, 1.6)

    for s in ["agents", "agents_cp", "tb"]:
        if not os.path.exists(s):
            os.makedirs(s)

        # Random ID of this session
        if config["default_session_ID"] is None:
            config["session_ID"] = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))
        else:
            config["session_ID"] = "TST"

    # Import correct env by name
    env_fun = my_utils.import_env(config["env_name"])
    env = env_fun(config)

    policy = my_utils.make_policy(env, config)

    if config["algo"] == "cma":
        train(env, policy, config)
    elif config["algo"] == "optuna":
        train_optuna(env, policy, config)
    else:
        print("Algorithm not implemented")
        exit()

    avg_episode_rew = test_agent(env, policy, config["N_test"], print_rew=False)

    env.close()
    del env

    return avg_episode_rew

if __name__ == "__main__":
    algo_config = my_utils.read_config("configs/es_default_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_hexapod/configs/eef.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 7
    config["verbose"] = -9
    config["save_agent"] = False
    config["animate"] = False
    config["tensorboard_log"] = False
    config["dummy_vec_env"] = False
    config["N_test"] = 30
    N_trials = 100

    t1 = time.time()
    study = optuna.create_study(direction='maximize', study_name="hexapod_eef_cyc_study", storage='sqlite:///hexapod_eef_cyc.db', load_if_exists=True)
    study.optimize(lambda x : objective(x, config), n_trials=N_trials, show_progress_bar=True)
    t2 = time.time()
    print("Time taken: ", t2-t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)





