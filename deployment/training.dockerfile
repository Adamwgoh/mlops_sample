FROM golang:alpine
RUN apk update && apk add --no-cache make protobuf-dev

FROM python:3.8.16
WORKDIR /
ADD requirements.txt requirements.txt
ADD common common
ADD Configs Configs
ADD Datasets Datasets
ADD db db
ADD deployment deployment
ADD scripts scripts
ADD running_scripts running_scripts
ADD Logs Logs
ADD Training Training
RUN pip install -r requirements.txt
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/savedmodel
# init tensorboard
# N nohup tensorboard --logdir ./Tensorboard &

# init mlflow ui
ENTRYPOINT ["python","Training/u2net_training_TF.py"]
