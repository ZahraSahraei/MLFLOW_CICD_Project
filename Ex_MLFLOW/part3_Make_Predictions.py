import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow.pyfunc


"""
#local server
mlflow_server_uri='http://127.0.0.1:5000/'
mlflow.set_tracking_uri(mlflow_server_uri)
#mlflow.get_experiment('iris Classifier Insights')

url= "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

data = pd.read_csv(url, sep=',' , header=None, names=columns)

# Separate features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column (species)
   
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)

run_id = '59374e9336cb46e8b1677e2f0bb68caf'  # Replace with your target Run ID

registered_model_name = "random forest, iris" 
model_stage = "Production"
version ="2"

# address model from Production stage
#logged_model = f"runs:/{run_id}/random forest, iris/{model_stage}"
logged_model = f"models:/{registered_model_name}/Production"


#Load the model as a PyFuncModel --- >use version 2 of model 
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Convert X_test to numpy.array to avoid the warning
X_test_array = X_test.to_numpy()

# Predict on the test set
predictions = loaded_model.predict(X_test_array)

# Evaluate the predictions

print(f"Predictions:\n{predictions}")

artifact_uri = mlflow.get_artifact_uri()
print(f"{artifact_uri}")

"""

"""
import mlflow

# Local MLflow server
mlflow_server_uri = 'http://127.0.0.1:5000/'
mlflow.set_tracking_uri(mlflow_server_uri)

# Define the Run ID you want to inspect
run_id = '59374e9336cb46e8b1677e2f0bb68caf'  # Replace with your target Run ID

# Retrieve the details of the run
run_info = mlflow.get_run(run_id)

# Print the artifact URI (location of the saved model or other artifacts)
artifact_uri = run_info.info.artifact_uri
print(f"Artifact URI for Run ID {run_id}: {artifact_uri}")

# Print additional information about the run (optional)
print(f"Run Parameters: {run_info.data.params}")
print(f"Run Metrics: {run_info.data.metrics}")
print(f"Run Tags: {run_info.data.tags}")
"""


"""
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient

# Local server URI
mlflow_server_uri = 'http://127.0.0.1:5000/'
mlflow.set_tracking_uri(mlflow_server_uri)

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(url, sep=',', header=None, names=columns)

# Split data
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model info
registered_model_name = "random forest, iris"
model_stage = "Production"  # Stage to load model from

# Create MlflowClient to check model details
client = MlflowClient()

# Check if model exists in the specified stage
try:
    model_versions = client.get_latest_versions(registered_model_name, stages=[model_stage])
    if not model_versions:
        raise Exception(f"No models found in stage '{model_stage}' for registered model '{registered_model_name}'.")

    # Print model details
    for version in model_versions:
        print(f"Model Name: {version.name}, Version: {version.version}, Stage: {version.current_stage}")

    # Use the first available model in the specified stage
    model_uri = f"models:/{registered_model_name}/{model_stage}"
    print(f"Loading model from URI: {model_uri}")
    
    # Load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print(f"Loaded model from URI: {loaded_model}")
    
    # Predict
    X_test_array = X_test.to_numpy()  # Convert to numpy array
    predictions = loaded_model.predict(X_test_array)
    print(f"Predictions:\n{predictions}")

except Exception as e:
    print(f"Error: {str(e)}")
    
"""


import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split

# Set the MLflow server URI
mlflow_server_uri = 'http://127.0.0.1:5000/'
mlflow.set_tracking_uri(mlflow_server_uri)

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(url, sep=',', header=None, names=columns)

#  Split dataset into training and testing sets
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column (species)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Information about the registered model
registered_model_name = "random forest, iris"
model_stage = "Production"

# Retrieve model versions using the newer API (replaces get_latest_versions)
client = MlflowClient()
model_versions = client.search_model_versions(f"name='{registered_model_name}'")

# Filter model versions that are in the specified stage (e.g., Production)
filtered_versions = [
    version for version in model_versions if version.current_stage == model_stage
]

# Raise an error if no models are found in the specified stage
if not filtered_versions:
    raise Exception(f"No models found in stage '{model_stage}' for registered model '{registered_model_name}'.")

#  Select the latest version of the model in the specified stage
latest_version = max(filtered_versions, key=lambda v: int(v.version))
model_uri = f"models:/{registered_model_name}/{model_stage}"

print(f"Model URI: {model_uri}")

# Load the model using the URI
loaded_model = mlflow.pyfunc.load_model(model_uri)
print(f"Model loaded successfully from URI: {model_uri}")

# Perform predictions with the loaded model
X_test_array = X_test.to_numpy()
predictions = loaded_model.predict(X_test_array)

#  Display the predictions
print(f"Predictions:\n{predictions}")

"""

#connect to artifact for test
import requests

artifact_url = "http://127.0.0.1:5000/experiments/135740183885560034/runs/59374e9336cb46e8b1677e2f0bb68caf/artifacts/"
response = requests.get(artifact_url)

if response.status_code == 200:
    print("Artifact successfully downloaded.")
else:
    print(f"Failed to download artifact. Status code: {response.status_code}")
    print(f"Error: {response.text}")

"""