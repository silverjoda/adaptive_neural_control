import optuna
import time

def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    time.sleep(0.01)
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize', study_name="example-study", storage='sqlite:///example.db', load_if_exists=True)
    study.optimize(objective, n_trials=10000)