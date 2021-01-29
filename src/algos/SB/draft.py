import optuna
from baselines3_RL import *
from copy import deepcopy
from multiprocessing import Pool, Process, Manager
import matplotlib.pyplot as plt

vals = []
def objective(trial):
    vals.append(trial.suggest_loguniform('gamma', 0.99, 0.999))
    return np.random.rand()

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    plt.hlines(1, 1, 20)  # Draw a horizontal line
    plt.xlim(0.99, 0.999)
    plt.ylim(0.5, 1.5)

    y = np.ones(np.shape(vals))  # Make all y values the same
    plt.plot(vals, y, '|', ms=40)  # Plot a line at each location specified in a
    plt.axis('off')
    plt.show()

    print("Best params: ", study.best_params, " Best value: ", study.best_value)





