FROM apache/airflow:2.5.3
COPY --chown=50000 requirements.txt .
COPY --chown=50000 car_prices.csv data/
RUN pip install --no-cache-dir -r requirements.txt