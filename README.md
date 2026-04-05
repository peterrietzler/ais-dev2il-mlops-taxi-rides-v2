# ais-dev2il-mlops-taxi-rides-v2


## Data Management with DVC 

- Prepare a repository that contains the raw data files 
- Download the raw data and store it in the data folder
- Use DVC to track the data 
- Push the data to a remote
- Create the 


fork my github repo 
connect the github repository with dagshub
start on the main branch
get the data from dvc 
dvc get https://github.com/peterrietzler/ais-dev2il-mlops-taxi-rides-v2 data
delete the all file
dvc init
dvc add ...
// setup dagshub connectivity 
dvc push -r ...

dagshub 
- settings -> integrations -> dvc


uv add --dev dvc
uv add --dev "dvc[s3]"


dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/peter.rietzler.privat/ais-dev2il-mlops-taxi-rides-v2.s3 
dvc remote modify origin --local access_key_id  XXX
dvc remote modify origin --local secret_access_key XXX

uvx dvc init

dvc config core.autostage true --> auto stage files in GIT
ucx dvc 

set remote to local file system
uvx dvc remote add -d local_folder /Users/peter.rietzler/Documents/private/dev2il/ais-dev2il/dvc-storage

 uvx dvc push


export MLFLOW_TRACKING_URI=https://dagshub.com/peter.rietzler.privat/ais-dev2il-mlops-taxi-rides-v2.mlflow
export MLFLOW_TRACKING_USERNAME=peter.rietzler.privat
export MLFLOW_TRACKING_PASSWORD=
uv run outlier_detector_training.py random_forest


# Workflow

New features: matrix

# TODOs

Training takes too long. Reduce the dataset for the training to make it faster.
