FROM ghcr.io/mlflow/mlflow:latest

USER root
RUN pip install psycopg2-binary
USER 1000
