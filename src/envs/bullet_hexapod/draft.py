import optuna
import time
import sqlite3
import sqlalchemy.exc
import numpy as np
def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    time.sleep(0.01)
    if np.random.rand() < 0.01:
        raise sqlite3.OperationalError
        #raise sqlalchemy.exc.InvalidRequestError
    return (x - 2) ** 2

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize', study_name="example-study", storage='sqlite:///example.db',
                                load_if_exists=True)
    #study = optuna.create_study()
    while True:
        try:
            study.optimize(objective, n_trials=10)
        except sqlite3.OperationalError:
            print("Excepted sqlite3.OperationalError")
        except sqlalchemy.exc.InvalidRequestError:
            print("Excepted sqlalchemy.exc.InvalidRequestError")
        except KeyboardInterrupt:
            print("Excepted Interrupt error")

