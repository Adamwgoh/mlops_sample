FROM golang:alpine
RUN apk update && apk add --no-cache make protobuf-dev

FROM python:3.8.16
WORKDIR /
ADD requirements.txt requirements.txt
ADD Logs Logs
ADD Model Model
ADD common common
ADD Configs Configs
ADD scripts scripts
ADD Training Training
ADD Datasets Datasets
ADD Tensorboard Tensorboard
ADD deployment deployment
ADD running_scripts running_scripts
RUN pip install -r requirements.txt
ENV MLFLOW_TRACKING_URI=http://50.1.1.4:5000
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/savedmodel
# init tensorboard
# N nohup tensorboard --logdir ./Tensorboard &

# init mlflow ui
ENTRYPOINT ["python","Training/u2net_training_TF.py"]
