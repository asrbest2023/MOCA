#!/bin/bash

# Step 1: Unzip the dataset.zip file at the current location
unzip Dataset.zip

# Step 2: Install Python packages from requirements.txt
pip install -r requirements.txt

# Step 3: Run finetune.py
python finetune.py
