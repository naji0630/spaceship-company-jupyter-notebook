#!/bin/bash

# Start MLflow UI in the background
mlflow ui --host 0.0.0.0 --port 5000 &

# Start Jupyter Notebook
start-notebook.sh