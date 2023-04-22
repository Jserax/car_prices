import os
from sqlalchemy import create_engine
from minio import Minio
import pandas as pd
import datetime as dt
from airflow import DAG
from airflow.operators.python import PythonOperator

POSTGRES_DB = os.environ['POSTGRES_DB']
POSTGRES_USER = os.environ['POSTGRES_USER']
POSTGRES_PASSWORD = os.environ['POSTGRES_PASSWORD']


def db_init():

    """Create buckets in minio and send data to postgresql"""

    client = Minio('minio',
                   access_key=os.environ['ACCESS_KEY'],
                   secret_key=os.environ['SECRET_KEY'],
                   secure=False)

    engine = create_engine((f'postgresql+psycopg2://{POSTGRES_USER}:'
                            f'{POSTGRES_PASSWORD}@postgresql/{POSTGRES_DB}'))

    client.make_bucket('data')
    client.make_bucket('mlflow')
    df = pd.read_csv('data/raw/car_prices.csv')

    timestamp = dt.datetime.now().timestamp()
    df.iloc[:750000]['timestamp'] = timestamp
    df.iloc[:750000].to_sql('car_prices', con=engine,
                            if_exists='replace', index=False)
    df.iloc[750000:].to_sql('requests', con=engine,
                            if_exists='replace', index=False)


with DAG(dag_id='prepare_data',
         start_date=dt.datetime(2000, 1, 1),
         description="Data preparation for model training",
         default_args={
            "depends_on_past": False,
            "retries": 1},
         schedule_interval=None,
         catchup=False) as dag:

    init_db_task = PythonOperator(
        python_callable=db_init, task_id='init_db')
