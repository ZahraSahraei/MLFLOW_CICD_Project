from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel
from typing import List
from mlflow.tracking import MlflowClient

# Set the MLflow tracking URI
mlflow.set_tracking_uri("https://mlflow.msinamsina.ir")

# Initialize the MLflow Client
client = MlflowClient()

# Define the model name
model_name = "IRIS"

# Fetch the latest version of the model in "Production" stage
latest_versions = client.get_latest_versions(model_name, ["Production"])
# Check if a version in "Production" exists
print(latest_versions)
if latest_versions:
    latest_version = latest_versions[0].version
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model '{model_name}' loaded from {model_uri}")
else:
    raise Exception(f"No model found in the 'Production' stage for {model_name}")

# Create a FastAPI app
app = FastAPI()

# Define the request body model using Pydantic
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: List[InputData]):
    # Convert input data to a pandas DataFrame
    input_data = pd.DataFrame([item.dict() for item in data])

    # Make predictions using the loaded model
    predictions = model.predict(input_data).tolist()

    for i in range(len(predictions)):
        if predictions[i] == 0:
            predictions[i] = 'setosa'
        elif predictions[i] == 1:
            predictions[i] = 'versicolor'
        else:
            predictions[i] = 'virginica'

    # Return predictions in the response
    return {"predictions": predictions}

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
