import optuna
from baselines3_RL import *
from copy import deepcopy
from multiprocessing import Pool, Process, Manager

def train_and_eval(x, procnum, return_dict):
    value = x ** 2 + np.random.randn() * 0.1
    return_dict[procnum] = value

def objective(trial):
    # Hexapod
    x = trial.suggest_uniform('x', -10, 10)
    res = {}

    N_reps = 8

    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(N_reps):
        p = Process(target=train_and_eval, args=(x, i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    return np.mean([k for k in return_dict.values()])

if __name__ == "__main__":
    t1 = time.time()
    #study = optuna.create_study(direction='minimize', study_name="example-study", storage='sqlite:///example.db', load_if_exists=True)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    print(time.time() - t1)

    print("Best params: ", study.best_params, " Best value: ", study.best_value)





