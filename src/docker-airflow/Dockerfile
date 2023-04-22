FROM apache/airflow:2.5.3
COPY --chown=50000 requirements.txt .
COPY --chown=50000 archive.zip data/
RUN unzip archive.zip
RUN rm archive.zip
RUN pip install --no-cache-dir -r requirements.txt