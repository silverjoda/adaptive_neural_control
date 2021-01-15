from baselines3_RL import *
from copy import deepcopy

def trial(config):
    env, model, _, stats_path = setup_train(config, setup_dirs=False)

    model.learn(total_timesteps=config["iters"])
    env.save(stats_path)

    eval_env = setup_eval(config, stats_path, seed=1337)
    model.set_env(eval_env)
    avg_episode_rew = test_agent(eval_env, model, deterministic=True, N=1, render=False, print_rew=False)

    env.close()
    eval_env.close()
    del env
    del eval_env
    del model

    return avg_episode_rew[0]

if __name__ == "__main__":
    env_fun = my_utils.import_env("quadrotor_stab")
    algo_config = my_utils.read_config("configs/a2c_quadrotor_config.yaml")
    env_config = my_utils.read_config("../../envs/bullet_quadrotor/configs/default.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 30000
    config["verbose"] = False
    config["animate"] = False
    config["default_session_ID"] = "OPT"

    pprint(config)

    # Specify variables
    var_dict = {"norm_obs" : [True, False], "learning_rate" : [1e-5, 5e-5, 1e-4, 3e-4, 7e-4]}

    results_dict = {}
    for varname, options_list in var_dict.items():
        for opt in options_list:
            trial_config = deepcopy(config)
            trial_config[varname] = opt
            print(f"Starting evaluation for {varname} = {opt}")
            avg_episode_rew = trial(trial_config)
            print(f"For option: {varname} = {opt}, avg_episode_rew attained: {avg_episode_rew}")
            results_dict[(varname, opt)] = avg_episode_rew

    print("Results: ")
    pprint(results_dict)






