import optuna
from baselines3_TD3 import *

def objective(trial, config):
    # Hexapod
    config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 1e-4, 5e-3))
    config["gamma"] = trial.suggest_loguniform('gamma', 0.96, 0.999)
    config["phase_scalar"] = trial.suggest_uniform('phase_scalar', .1, .5)
    config["z_aux_scalar"] = trial.suggest_uniform('z_aux_scalar', 0.00, 0.04)
    config["x_mult_scalar"] = trial.suggest_uniform('x_mult_scalar', 0.00, 0.02)
    config["ou_sigma"] = trial.suggest_uniform('ou_sigma', 0.1, 0.8)
    config["batchsize"] = trial.suggest_int('batchsize', 32, 512)

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
        model.save("agents/EEF_TD3_OPTUNA_policy")
        print("Saved best policy")

    env.close()
    eval_env.close()
    del env
    del eval_env
    del model

    return avg_episode_rew

if __name__ == "__main__":
    algo_config = my_utils.read_config("configs/td3_default_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_hexapod/configs/eef.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 200000
    config["verbose"] = False
    config["animate"] = False
    config["tensorboard_log"] = False
    config["dummy_vec_env"] = False
    config["N_test"] = 30
    N_trials = 100

    t1 = time.time()
    study = optuna.create_study(direction='maximize', study_name="hexapod_eef_opt_study", storage='sqlite:///hexapod_eef_opt.db', load_if_exists=True)
    #study = optuna.create_study(direction='maximize')
    study.optimize(lambda x : objective(x, config), n_trials=N_trials, show_progress_bar=True)
    t2 = time.time()
    print("Time taken: ", t2-t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)





