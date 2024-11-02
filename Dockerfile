# Use jupyter/scipy-notebook as the base image
FROM gcr.io/kaggle-gpu-images/python

# Switch to root user
USER root

# Install additional packages
RUN pip install --no-cache-dir scikit-learn xgboost lightgbm catboost mlflow optuna

# Copy specific data into the image
COPY data /home/jovyan/data

# Copy the start.sh script into the image
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Revert back to the jovyan user
USER $NB_UID

# Set the entrypoint to start.sh
ENTRYPOINT ["/usr/local/bin/start.sh"]
