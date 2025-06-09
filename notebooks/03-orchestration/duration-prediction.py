import gzip
import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import root_mean_squared_error

import mlflow
from pathlib import Path

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('nyc-taxi-experiment')

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    if url.endswith('.csv'):
        df = pd.read_csv(url)

        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    elif url.endswith('.parquet'):
        df = pd.read_parquet(url)
        print(f"Number of records loaded: {len(df)}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    return df


df_train = read_dataframe(year=2023, month=3)
print(f"Size of the result: {len(df_train)} rows")
df_val = read_dataframe(year=2023, month=2)

def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv



target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values


models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        mlflow.set_tag("developer", "ananya")
        mlflow.log_param("train_data_path", "./data/yellow_tripdata_2023-01.parquet")
        mlflow.log_param("val_data_path", "./data/yellow_tripdata_2023-02.parquet")

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)

        mlflow.log_metric("rmse", rmse)
        print("Intercept:", lr.intercept_)

        # Save and log DictVectorizer (compressed)
        dv_path = "models/dv.pkl.gz"
        with gzip.open(dv_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(dv_path, artifact_path="preprocessor")

        # Save and log the trained model
        mlflow.sklearn.log_model(lr, artifact_path="model")

        return run.info.run_id


def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open("run_id.txt", "w") as f:
        f.write(run_id)

