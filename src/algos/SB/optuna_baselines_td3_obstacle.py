import optuna
from baselines3_TD3 import *

def objective(trial, config):
    # Hexapod
    config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 7e-4, 3e-3))
    config["gamma"] = trial.suggest_loguniform('gamma', 0.97, 0.99)
    config["ou_sigma"] = trial.suggest_uniform('ou_sigma', 0.5, 1.5)
    jrl = [-0.6,
           trial.suggest_uniform('jrl_femur', -1.5, -0.8),
           trial.suggest_uniform('jrl_tibia', -0.8, -0.2)]
    jr_diff = [1.2,
           trial.suggest_uniform('jr_diff_femur', 1, 1.6),
           trial.suggest_uniform('jr_diff_tibia', 1.2, 2.0)]

    config["joints_rads_low"] = jrl
    config["joints_rads_diff"] = [jrl[i]+jr_diff[i] for i in range(3)]
    config["joints_rads_high"] = [jrl[i]+jr_diff[i] for i in range(3)]

    # Quad
    # config["n_steps"] = trial.suggest_int('n_steps', 6, 40)
    # config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 1e-4, 7e-4))
    # config["gamma"] = trial.suggest_loguniform('gamma', 0.985, 0.999)
    # config["policy_hid_dim"] = trial.suggest_int("policy_hid_dim", 48, 256)

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
        model.save("agents/OBSTACLE_TD3_OPTUNA_policy")
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
    config["iters"] = 300000
    config["verbose"] = False
    config["animate"] = False
    #config["default_session_ID"] = "OPT_HEX"
    config["tensorboard_log"] = False
    config["dummy_vec_env"] = False
    config["N_test"] = 30
    N_trials = 70

    t1 = time.time()
    study = optuna.create_study(direction='maximize', study_name="hexapod_opt_study", storage='sqlite:///hexapod_opt.db', load_if_exists=True)
    #study = optuna.create_study(direction='maximize')
    study.optimize(lambda x : objective(x, config), n_trials=N_trials, show_progress_bar=True)
    t2 = time.time()
    print("Time taken: ", t2-t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)





