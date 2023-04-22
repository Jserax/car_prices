import os
import pandas as pd
import datetime as dt
import mlflow
import bentoml
from sqlalchemy import create_engine
from minio import Minio
from io import BytesIO
from catboost import CatBoostClassifier

import optuna
from optuna.integration.mlflow import MLflowCallback

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from evidently import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

POSTGRES_DB = os.environ['POSTGRES_DB']
POSTGRES_USER = os.environ['POSTGRES_USER']
POSTGRES_PASSWORD = os.environ['POSTGRES_PASSWORD']
CAT_COLS = ['brand', 'name', 'bodyType', 'color',
            'fuelType', 'transmission', 'location']
NUM_COLS = ['year', 'mileage', 'power', 'engineDisplacement']
TARGET = 'price'
MLFLOW_URI = f"mlflow:{os.environ['MLFLOW_PORT']}"
EXPERIMENT_NAME = 'car_price'


def check_data_drift() -> str:

    """Check data drift"""

    column_mapping = ColumnMapping()

    column_mapping.target = TARGET
    column_mapping.categorical_features = CAT_COLS
    column_mapping.numerical_features = NUM_COLS

    client = Minio('minio',
                   access_key=os.environ['ACCESS_KEY'],
                   secret_key=os.environ['SECRET_KEY'],
                   secure=False)
    train_set = Variable.get('last_train_set', None)
    if train_set is None:
        return "prepare_features"
    engine = create_engine((f'postgresql+psycopg2://{POSTGRES_USER}:'
                            f'{POSTGRES_PASSWORD}@postgresql/{POSTGRES_DB}'))
    last_date = Variable.get('last_date', 1000000000)
    current_date = dt.datetime.now()
    query = (f"SELECT * FROM car_prices WHERE timestamp >= {last_date}"
             f" AND timestamp < {current_date.timestamp()}")
    df = pd.read_sql(query, con=engine, index_col=0)
    df['engineDisplacement'] = df['engineDisplacement'].str.split(' ') \
        .str[0].astype('float32')
    train_set = Variable.get('last_train_set')
    obj = client.get_object(
        "data",
        train_set)
    ref = pd.read_csv(obj)
    data_drift = TestSuite(tests=[DataDriftTestPreset()])
    data_drift.run(reference_data=ref, current_data=df)
    if data_drift.as_dict()['tests'][0]['status'] == "FAIL":
        data_drift.save_html('data_drift.html')
        return "prepare_features"
    return "end_dag"


def prepare_features() -> None:

    """Load data from postgresql, process, split and save dataset to Minio"""

    client = Minio('minio',
                   access_key=os.environ['ACCESS_KEY'],
                   secret_key=os.environ['SECRET_KEY'],
                   secure=False)

    engine = create_engine((f'postgresql+psycopg2://{POSTGRES_USER}:'
                            f'{POSTGRES_PASSWORD}@postgresql/{POSTGRES_DB}'))
    last_date = Variable.get('last_date', 1000000000)
    current_date = dt.datetime.now()
    query = (f"SELECT * FROM car_prices WHERE timestamp >= {last_date}"
             f" AND timestamp < {current_date.timestamp()}")
    df = pd.read_sql(query, con=engine, index_col=0)
    df['engineDisplacement'] = df['engineDisplacement'].str.split(' ') \
        .str[0].astype('float32')
    df[CAT_COLS] = df[CAT_COLS].astype('category')
    train, test = train_test_split(df, test_size=0.2)
    train_set = f'train_{current_date.date()}-{current_date.hour}.csv'
    test_set = f'test_{current_date.date()}-{current_date.hour}.csv'
    Variable.set('last_train_set', train_set)
    Variable.set('last_test_set', test_set)
    train = train.to_csv().encode('utf-8')
    client.put_object(
        "data",
        train_set,
        data=BytesIO(train),
        length=len(train),
        content_type='application/csv')
    test = test.to_csv().encode('utf-8')
    client.put_object(
        "data",
        test_set,
        data=BytesIO(test),
        length=len(test),
        content_type='application/csv')


def objective(trial, x_train, y_train):

    iter = trial.suggest_int("iterations", 100, 1500)
    lr = trial.suggest_float("learning_rate", 1e-4, 1, log=True)
    depth = trial.suggest_int("depth", 2, 10)
    numeric_preprocessor = Pipeline(
        [("imputation_mean", SimpleImputer(strategy='median')),
         ("scaler", StandardScaler())])
    categorical_preprocessor = Pipeline(
        [("imputation_most_freq", SimpleImputer(strategy='most_frequent')),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        [("categorical", categorical_preprocessor, CAT_COLS),
         ("numerical", numeric_preprocessor, NUM_COLS)])

    pipe = Pipeline([('preprocessor', preprocessor),
                    ('catboost', CatBoostClassifier(iterations=iter,
                                                    depth=depth,
                                                    learning_rate=lr,
                                                    verbose=False))])
    mse = cross_val_score(pipe, x_train, y_train,
                          cv=3, scoring='neg_mean_squared_error').mean()
    return mse


def train_model() -> None:

    """Train model"""

    client = Minio('minio',
                   access_key=os.environ['ACCESS_KEY'],
                   secret_key=os.environ['SECRET_KEY'],
                   secure=False)
    train_set = Variable.get('last_train_set')
    test_set = Variable.get('last_test_set')
    obj = client.get_object(
        "data",
        train_set)
    train = pd.read_csv(obj)
    obj = client.get_object(
        "data",
        test_set)
    test = pd.read_csv(obj)
    x_train = train[NUM_COLS+CAT_COLS]
    y_train = train[TARGET]
    x_test = test[NUM_COLS+CAT_COLS]
    y_test = test[TARGET]
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflc = MLflowCallback(metric_name="f1_weighted",
                           tracking_uri=MLFLOW_URI,
                           mlflow_kwargs={"tags": {'type': 'train'}})
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="car_price",
                                sampler=sampler,
                                direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, y_train),
                   n_trials=5, callbacks=[mlflc])
    numeric_preprocessor = Pipeline(
        [("imputation_mean", SimpleImputer(strategy='median')),
         ("scaler", StandardScaler())])
    categorical_preprocessor = Pipeline(
        [("imputation_most_freq", SimpleImputer(strategy='most_frequent')),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        [("categorical", categorical_preprocessor, CAT_COLS),
         ("numerical", numeric_preprocessor, NUM_COLS)])

    pipe = Pipeline([('preprocessor', preprocessor),
                    ('catboost', CatBoostClassifier(verbose=False,
                                                    **study.best_params))])

    with mlflow.start_run(tags={'type': 'test'}):
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_params(**study.best_params)
        mlflow.log_metrics({'mse': mse})
        mlflow.log_artifact('data_drift.html')
        model = mlflow.sklearn.log_model(pipe, "model",
                                         registered_model_name="car_price")
        model_uri = model.model_uri
        bentoml.mlflow.import_model('car_price', model_uri)


with DAG(dag_id='prepare_data',
         start_date=dt.datetime(2000, 1, 1),
         description="Data preparation for model training",
         default_args={
            "depends_on_past": False,
            "retries": 1},
         schedule_interval="@hourly",
         catchup=False,
         tags=["critical", "data"]) as dag:

    start_dag = EmptyOperator(
        task_id='start_dag')

    end_dag = EmptyOperator(
        task_id='end_dag')

    check_drift_task = BranchPythonOperator(
        python_callable=check_data_drift, task_id="check_data_drift")

    prepare_features_task = PythonOperator(
        python_callable=prepare_features, task_id="prepare_features")

    train_model_task = PythonOperator(
        python_callable=train_model, task_id="train_model")

    build_model_task = BashOperator(
        bash_command=("cd /root/bentoml && bentoml build && bentoml "
                      "containerize car_price:latest -t car_price:latest"),
        task_id="build_model")

    run_model_task = BashOperator(
        bash_command="docker run -d -p 3000:3000",
        task_id="run_model")

    start_dag >> check_drift_task >> [prepare_features_task, end_dag]
    prepare_features_task >> train_model_task >> build_model_task
    build_model_task >> run_model_task >> end_dag
