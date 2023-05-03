# docker run -p 5000:5000 mlflow_server
FROM python:3.8
WORKDIR /mlflow

ADD deployment/mlflow-server-requirements.txt requirements.txt
ADD db/common db/common
RUN pip install --no-cache-dir -r requirements.txt
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/savedmodel
EXPOSE 5000
RUN apt update
RUN apt install sqlite3
CMD mlflow server \
    --default-artifact-root $PWD/savedmodel \
    --backend-store-uri sqlite:///db/mlruns.db \
    --registry-store-uri sqlite:///db/registered_model.db \
    --host 0.0.0.0 \
    --port 5000