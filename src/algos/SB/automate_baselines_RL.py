from baselines3_RL import *
from copy import deepcopy

def trial(config):
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

    # env_fun = my_utils.import_env("hexapod_wp_eef")
    # algo_config = my_utils.read_config("configs/a2c_hexapod_config.yaml")
    # env_config = my_utils.read_config("../../envs/bullet_hexapod/configs/eef.yaml")

    config = {**algo_config, **env_config}
    config["iters"] = 30000000
    config["verbose"] = False
    config["animate"] = False
    config["animate"] = False
    config["default_session_ID"] = "OPT_QUAD"
    config["tensorboard_log"] = False
    N_reps = 1

    pprint(config)

    # Specify variables
    var_dict_list = [{"output_transport_delay" : i, "act_input" : i, "obs_input" : i} for i in range(10)]
    # var_dict_list = [{"norm_obs" : False, "norm_reward" : False, "w_1" : False, "w_2" : True},
    #                  {"norm_obs" : True, "norm_reward" : False, "w_1" : False, "w_2" : True},
    #                  {"norm_obs" : False, "norm_reward" : True, "w_1" : False, "w_2" : True},
    #                  {"norm_obs" : True, "norm_reward" : True, "w_1" : False, "w_2" : True},
    #                  {"norm_obs" : True, "norm_reward" : True, "w_1" : True, "w_2" : False}]

    results_list = []
    best_result_dict = {}
    best_result_value = -1e10

    for var_dict in var_dict_list:
        mean_avg_episode_rew = 0
        for _ in range(N_reps):
            trial_config = deepcopy(config)
            for varname, opt in var_dict.items():
                trial_config[varname] = opt
            print(f"Starting evaluation for dict: {var_dict}")
            avg_episode_rew = trial(trial_config)
            mean_avg_episode_rew += avg_episode_rew
        mean_avg_episode_rew /= N_reps
        print(f"For dict:{var_dict}, avg_episode_rew attained: {mean_avg_episode_rew}")
        results_list.append((var_dict, mean_avg_episode_rew))
        if mean_avg_episode_rew > best_result_value:
            best_result_value = mean_avg_episode_rew
            best_result_dict = var_dict

    print("Results: ")
    pprint(results_list)

    print("Best result: ")
    print(best_result_dict, best_result_value)






