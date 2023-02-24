## Sample MLFlow workflow
This is a sample implementation of MLFlow on a tensorflow training workflow, where its logs is stored in a sql database. As I intend to build upon this to be a full fledged mlops workflow, this readme will change and the purpose of this repo will ultimately change as well

# Intro

## Requirements
* Tensorflow
* Tensorflow dataset
* Mlflow
* Tensorboard 
* Matplotlib

## Quick Go-to
`pip install -r requirements.txt`
`pip install -e .`
`. scripts/init_mlflow.sh`

## Future add-ons and timeline:
* Adding CVAT and Fiftyone for dataset management and annotation capabilities
* Adding REST API to mlflow for external deployments
* Adding model comparisons and auto-stage changes and deployments
* Adding model serving via REST API
* Adding CI and test cases