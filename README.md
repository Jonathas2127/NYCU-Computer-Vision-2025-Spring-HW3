# NYCU-Computer-Vision-2025-Spring-HW3
student ID: 111550169

Name: 陳宣澔

## Introduction
This project implements an instance segmentation pipeline for microscopic cell images using Mask R-CNN. The model is built on top of PyTorch's torchvision library and uses a ResNet-50 FPN backbone to detect and segment individual cell instances across four classes. The input images and ground-truth masks are preprocessed with Albumentations, and training is conducted over 50 epochs with a focus on optimizing segmentation quality and generalization. Optional extensions include center and boundary map supervision for enhanced spatial understanding.

## How to install
I run train.py on Kaggle Notebooks.
Upload datasets to Kaggle first and open a new notebook.

#First cell:

!pip install imagecodecs --quiet

#Second cell:

train.py

#Third cell (add secrets with API):

import os
import json
import shutil
from kaggle_secrets import UserSecretsClient

os.makedirs('/root/.kaggle', exist_ok=True)
user_secrets = UserSecretsClient()
kaggle_json = user_secrets.get_secret("kaggle_json")

with open('/root/.kaggle/kaggle.json', 'w') as f:
    f.write(kaggle_json)
os.chmod('/root/.kaggle/kaggle.json', 0o600)

dataset_dir = '/kaggle/working/dataset'
os.makedirs(dataset_dir, exist_ok=True)
shutil.move('/kaggle/working/model_final.pth', dataset_dir)

dataset_metadata = {
    "title": "digit-recognition-output",
    "id": "Kaggle-user-name/model-path-output",  # Replace with your Kaggle account
    "licenses": [{"name": "CC0-1.0"}]
}

with open(f"{dataset_dir}/dataset-metadata.json", "w") as f:
    json.dump(dataset_metadata, f)

!pip install kaggle --quiet
!kaggle datasets create -p {dataset_dir} --dir-mode zip

Run all cells and the model path would be automatically saved in dataset.

Then, I run inference.py on PC with the model path downloading from the training section on Kaggle.

## Performance snapshot
[Snapshot on leaderboard](https://imgur.com/a/urPmtsm)
