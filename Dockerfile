FROM ghcr.io/mlflow/mlflow:latest

USER root
RUN pip install pymysql
USER 1000
