import mlflow
import mlflow.sklearn
import os
import platform
import psutil
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Global Parameters
def get_config():
    return {
        "learning_rate": 0.01,
        "batch_size": 32,
        "n_estimators": 10,
        "max_depth": 3
    }


# Define training function
def train_model(config):
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                        test_size=0.2,
                                                        random_state=42)

    model = RandomForestClassifier(n_estimators=config["n_estimators"],
                                   max_depth=config["max_depth"],
                                   random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions,
                                average='macro')  # Weighted for multi-class
    recall = recall_score(y_test, predictions,
                          average='weighted')  # Weighted for multi-class
    f1 = f1_score(y_test, predictions, average='weighted')

    return model, accuracy, predictions, y_test, precision, recall, f1


# Define logging function
def log_model_mlflow(config, model, accuracy,predictions, y_test, precision,recall, f1):
    mlflow.set_tracking_uri(uri='https://mlflow.msinamsina.ir/')
    mlflow.set_experiment('iris Classifier Insights')

    with mlflow.start_run():
        # Log parameters
        for param_name, param_value in config.items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1', f1)
        loss = np.mean(np.square(y_test - predictions))  # Mean Squared Error
        mlflow.log_metric('loss', loss)

        # Log system information
        mlflow.log_param("system_os", platform.system())
        mlflow.log_param("system_processor", platform.processor())
        mlflow.log_param("system_ram", f"{psutil.virtual_memory().total / 1e9:.2f} GB")
        mlflow.log_param("cpu_count", os.cpu_count())
        mlflow.log_param("python_version", platform.python_version())

        # Log detailed hardware information
        cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
        disk_usage = psutil.disk_usage('/').total / 1e9  # Disk size in GB
        mlflow.log_param("cpu_max_frequency_MHz", cpu_freq)
        mlflow.log_param("disk_total_GB", f"{disk_usage:.2f}")

        # Measure resource usage
        start_time = time.time()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1e6  # Memory in MB

        runtime = time.time() - start_time
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1e6  # Memory in MB
        memory_used = final_memory - initial_memory

        mlflow.log_metric("runtime", runtime)
        mlflow.log_metric("memory_used_MB", memory_used)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")


# Main Execution
if __name__ == "__main__":
    config = get_config()
    model, accuracy, predictions, y_test, precision, recall, f1 = train_model(config)
    log_model_mlflow(config, model, accuracy, predictions, y_test, precision, recall, f1)