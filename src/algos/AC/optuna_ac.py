
import optuna
from ac import *

if __name__ == "__main__":
    env_fun = my_utils.import_env("quadrotor_stab")
    args = parse_args()

    algo_config = my_utils.read_config(args["algo_config"])
    env_config = my_utils.read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    config["n_total_steps_train"] = 700000
    config["save_policy"] = False

    def objective(trial):
        config["batchsize"] = trial.suggest_categorical('batchsize', [4,8,16,32,64,128,256])
        config["init_log_std_value"] = trial.suggest_uniform("init_log_std_value", -1., 0)
        config["policy_learning_rate"] = trial.suggest_loguniform('policy_learning_rate', 1e-5, 1e-3)
        config["vf_learning_rate"] = trial.suggest_loguniform('vf_learning_rate', 1e-5, 1e-3)
        config["gamma"] = trial.suggest_loguniform('gamma', 0.93, 0.999)
        config["policy_grad_clip_value"] = trial.suggest_uniform('policy_grad_clip_value', 0.3, 0.9)

        env, policy, vf = setup_train(config)

        train(env, policy, vf, config)
        value = test_agent(env, policy, N=50, print_rew=False)
        env.close()
        del env

        return value

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)

    [print("---------------------------------") for _ in range(10)]
    print("Best params: ", study.best_params, " Best value: ", study.best_value)
    [print("---------------------------------") for _ in range(10)]




