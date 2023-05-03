## Sample MLFlow workflow
This is a sample implementation of MLFlow on a tensorflow training workflow, where its logs is stored in a sql database. As I intend to build upon this to be a full fledged mlops workflow, this readme will change and the purpose of this repo will ultimately change as well

# Quickstart
- Make sure mlflow server is running by `mlflow server`
- Entrypoint with `Training/u2net_trainig_TF.py
- You may use `scripts/init_mlflow.sh` to run both

## Requirements
* tensorflow == 2.10.1
* CUDA version == 11.2
* CuDNN version == 8.6.*
* Tensorflow dataset
* Mlflow == 2.1.1
* Tensorboard  == 2.10.1
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

## 3 May 2023 Update
* Dockerized mlflow server and training sequence. To run them, now run `docker compose -f deployment/docker-compose.yaml up -d'
* mlflow server will use 51.0.0.4:5000 as its internal IP and port.

## 26 April 2023 Update
* Terraform is now supported for kubernetes deployment. Make sure you have terraform installed [see how here](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)