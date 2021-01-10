from baselines_RL import *
from copy import deepcopy
if __name__ == "__main__":
    env_fun = my_utils.import_env("hexapod_wp")
    args = parse_args()

    algo_config = my_utils.read_config(args["algo_config"])
    env_config = my_utils.read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    def trial(config):
        env, model, _ , stats_path = setup_train(config, setup_dirs=False)

        model.learn(total_timesteps=config["iters"])

        env.training = False
        env.norm_reward = False

        # TODO: Make saving of env stats, and creation of new for evaluations

        avg_episode_rew = test_multiple(env, model, deterministic=True, N=100)

        env.close()
        del env
        del model

        return avg_episode_rew

    config["iters"] = 2000000

    # Specify variables
    var_dict = {"norm_obs" : [True, False], "learning_rate" : [2e-4, 7e-4]}

    best_avg_episode_rew = -1000
    best_partial_config = {}
    for varname, options_list in var_dict.items():
        for opt in options_list:
            modified_config = deepcopy(config)
            modified_config[varname] = opt
            avg_episode_rew = trial(modified_config)
            print("For config: ")
            pprint(modified_config)
            print(f"Avg_episode_rew attained: {avg_episode_rew}")
            if avg_episode_rew > best_avg_episode_rew:
                best_avg_episode_rew = avg_episode_rew
                best_partial_config = [varname, opt]

    print("For config: ")
    pprint(best_partial_config)
    print(f"Best avg_episode_rew attained: {best_avg_episode_rew}")








