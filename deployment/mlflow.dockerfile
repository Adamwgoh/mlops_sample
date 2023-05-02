FROM python:3.8.2
ADD common common
RUN python3 -c "import common"
COPY . /
RUN python3 -c "import common"
ADD requirements.txt requirements.txt
ADD Configs Configs
ADD Datasets Datasets
ADD db db
ADD deployment deployment
ADD scripts scripts
ADD running_scripts running_scripts
ADD Logs Logs
ADD Training Training
RUN cd /
RUN pip install -r requirements.txt
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/savedmodel
EXPOSE 5000
#ENTRYPOINT ["./scripts/init_mlflow.sh"]
RUN nohup mlflow server \
    --backend-store-uri sqlite:///db/mlruns.db \
    --default-artifact-root $PWD/savedmodel \
    --registry-store-uri sqlite:///db/registered_model.db \
    > Logs/mlflowerror.out &

# init tensorboard
RUN nohup tensorboard --logdir ./Tensorboard &

# init mlflow ui
RUN python Training/u2net_training_TF.py
