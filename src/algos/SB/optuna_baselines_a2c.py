#import warnings
#warnings.filterwarnings("ignore")
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import optuna
from baselines_RL import *

if __name__ == "__main__":
    env_fun = my_utils.import_env("quadrotor_stab")
    args = parse_args()

    algo_config = my_utils.read_config(args["algo_config"])
    env_config = my_utils.read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    config["iters"] = 1000000

    def objective(trial):
        config["n_steps"] = trial.suggest_int('n_steps', 10, 200)
        config["ent_coef"] = trial.suggest_loguniform("ent_coef", 0.000001, 0.001)
        config["learning_rate"] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        config["gamma"] = trial.suggest_loguniform('gamma', 0.93, 0.999)
        config["max_grad_norm"] = trial.suggest_uniform('max_grad_norm', 0.3, 0.8)
        config["vf_coef"] = trial.suggest_uniform('vf_coef', 0.3, 0.7)

        env, model, _ = setup_train(config, setup_dirs=False)

        model.learn(total_timesteps=config["iters"])
        #env.close()
        #del env

        #env_fun = my_utils.import_env(env_config["env_name"])
        #env = env_fun(config)
        #model.env = env
        value = test_multiple(env, model, deterministic=True, N=50, print_rew=False)

        env.close()
        del env
        del model

        return value


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    [print("---------------------------------") for _ in range(10)]
    print("Best params: ", study.best_params, " Best value: ", study.best_value)
    [print("---------------------------------") for _ in range(10)]




