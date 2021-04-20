import time
import pickle
import optuna
from train_forward_model import *
import sqlite3
import sqlalchemy.exc

def objective(trial, config):
    config["hidden_dim"] = trial.suggest_int('hidden_dim', 8, 128)
    config["non_linearity"] = trial.suggest_categorical("non_linearity",
                                                        ["nn.ReLU", "nn.ReLU6", "nn.LeakyReLU", "nn.ELU", "nn.Tanh"])
    config["learning_rate"] = trial.suggest_loguniform("learning_rate", 0.0001, 0.03)
    config["weight_decay"] = trial.suggest_loguniform("weight_decay", 0.00001, 0.01)
    config["trn_batchsize"] = trial.suggest_int("trn_batchsize", 8, 1024)

    fm = ForwardModelTrainer(config)
    fm.load_data()
    fm.train()
    score = fm.eval()

    try:
        best_value = trial.study.best_value
    except:
        best_value = -1e5
    if score > best_value:
        fm.save_model("opt_model")
        print("Saved best policy")
        pickle.dump(config, open( "models/buggy_model_best_params.p", "wb" ))

    del fm
    return score

if __name__ == "__main__":
    with open("configs/model_training.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["n_trn_iters"] = 50000
    config["verbose"] = False
    N_trials = 300

    t1 = time.time()
    study = optuna.create_study(direction='minimize', study_name="buggy_model_training_study", storage='sqlite:///buggy_model_training_opt.db', load_if_exists=True)

    while True:
        try:
            study.optimize(lambda x : objective(x, config), n_trials=N_trials, show_progress_bar=True)
            break
        except (sqlite3.OperationalError, sqlalchemy.exc.InvalidRequestError):
            print("Optimize failed, restarting")

    t2 = time.time()
    print("Time taken: ", t2-t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)





