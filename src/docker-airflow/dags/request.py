import requests
import os
import pandas as pd
import datetime as dt
from sqlalchemy import create_engine
from airflow import DAG
from airflow.operators.python import PythonOperator

POSTGRES_DB = os.environ['POSTGRES_DB']
POSTGRES_USER = os.environ['POSTGRES_USER']
POSTGRES_PASSWORD = os.environ['POSTGRES_PASSWORD']


def make_request(row) -> None:
    requests.post("http://model:3000/predict",
                  headers={"content-type": "application/json"},
                  data=row.to_json()).text


def make_requests() -> None:

    engine = create_engine((f'postgresql+psycopg2://{POSTGRES_USER}:'
                            f'{POSTGRES_PASSWORD}@postgresql/{POSTGRES_DB}'))
    df = pd.read_sql('SELECT * FROM requests', con=engine, index_col=[0])
    df.apply(make_request)


with DAG(dag_id='prepare_data',
         start_date=dt.datetime(2000, 1, 1),
         description="Data preparation for model training",
         default_args={
            "depends_on_past": False,
            "retries": 1},
         schedule_interval=None,
         catchup=False) as dag:

    make_requets_task = PythonOperator(
        python_callable=make_requests, task_id='make_request_to_model')
