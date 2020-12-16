#import warnings
#warnings.filterwarnings("ignore")
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import optuna
import time
from baselines_RL import *

if __name__ == "__main__":
    env_fun = my_utils.import_env("quadrotor_stab")
    args = parse_args()

    algo_config = my_utils.read_config(args["algo_config"])
    env_config = my_utils.read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    config["iters"] = 700000
    config["save_policy"] = False

    def objective(trial):
        config["batchsize"] = trial.suggest_categorical('batchsize', [4,8,16,32,64,128,256])
        config["init_log_std_value"] = trial.suggest_uniform("init_log_std_value", -1., 0)
        config["policy_learning_rate"] = trial.suggest_loguniform('policy_learning_rate', 1e-5, 1e-3)
        config["vf_learning_rate"] = trial.suggest_loguniform('vf_learning_rate', 1e-5, 1e-3)
        config["gamma"] = trial.suggest_loguniform('gamma', 0.93, 0.999)
        config["policy_grad_clip_value"] = trial.suggest_uniform('policy_grad_clip_value', 0.3, 0.9)

        env, model, _ = setup_train(config)

        model.learn(total_timesteps=config["iters"])
        env.close()
        del env

        env_fun = my_utils.import_env(env_config["env_name"])
        env = env_fun(config)
        model.env = env
        value = test_agent(env, model, deterministic=True, N=50, print_rew=False)

        env.close()
        del env
        del model

        return value

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    time.sleep(0.1); print(study.best_params, study.best_value)




