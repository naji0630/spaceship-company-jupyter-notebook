# Use jupyter/scipy-notebook as the base image
FROM jupyter/scipy-notebook

# Switch to root user
USER root

# Install additional packages
RUN pip install --no-cache-dir scikit-learn xgboost lightgbm catboost mlflow optuna wandb

# Copy specific data into the image
# Ensure there is a 'data/' folder in the same directory as this Dockerfile
COPY data /home/jovyan/data

# Revert back to the jovyan user
USER $NB_UID
