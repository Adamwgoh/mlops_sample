export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_DEFAULT_ARTIFACT_ROOT=./savedmodel


nohup mlflow server \
    --backend-store-uri sqlite:///db/mlruns.db \
    --default-artifact-root $PWD/savedmodel \
    --registry-store-uri sqlite:///db/registered_model.db \
    > Logs/mlflowerror.out &

# init tensorboard
nohup tensorboard --logdir ./Tensorboard &

# init mlflow ui
python Training/u2net_training_TF.py