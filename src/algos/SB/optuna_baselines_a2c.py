import optuna
from baselines3_RL import *
from copy import deepcopy

def objective(trial):
    config["n_steps"] = trial.suggest_int('n_steps', 6, 80)
    config["learning_rate"] = trial.suggest_loguniform('learning_rate', 1e-5, 4e-4)
    config["norm_reward"] = trial.suggest_categorical('norm_reward', [True, False])

    env, model, _, stats_path = setup_train(config, setup_dirs=False)
    model.learn(total_timesteps=config["iters"])
    env.save(stats_path)

    eval_env = setup_eval(config, stats_path, seed=1337)
    model.set_env(eval_env)
    N_test = 30
    avg_episode_rew = test_agent(eval_env, model, deterministic=True, N=N_test, render=False, print_rew=False)

    env.close()
    eval_env.close()
    del env
    del eval_env
    del model

    return avg_episode_rew[0] / N_test

if __name__ == "__main__":
    env_fun = my_utils.import_env("quadrotor_stab")
    algo_config = my_utils.read_config("configs/a2c_quadrotor_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_quadrotor/configs/default.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 30000
    config["verbose"] = False
    config["animate"] = False
    config["default_session_ID"] = "OPT"

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("Best params: ", study.best_params, " Best value: ", study.best_value)





