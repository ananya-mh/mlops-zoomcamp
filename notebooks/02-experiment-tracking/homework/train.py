import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('nyc-taxi-experiment')


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.set_tag("developer", "ananya")
        # mlflow.log_param("train_data_path", "./output/train.pkl")
        # mlflow.log_param("val_data_path", "./output/val.pkl")
        # mlflow.log_param("max_depth", 10)
        # mlflow.log_param("random_state", 0)
        # mlflow.log_param("model_type", "RandomForestRegressor")

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)

        model_path = "../models/rf_model.bin"
        with open(model_path, "wb") as f_out:
            pickle.dump(rf, f_out)

        mlflow.log_artifact(model_path, artifact_path="artifacts")


if __name__ == '__main__':
    run_train()
