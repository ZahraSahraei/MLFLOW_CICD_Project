import requests
import pandas as pd
import time

# Step 1: Load the dataset from the URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(url, sep=',', header=0, names=columns)
data = data[:5]  # For this example, we'll use only the first 5 samples

# Step 2: Prepare the input data for the model (exclude the 'species' column for prediction)
features = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Step 3: Define the API endpoint
api_url = "http://localhost:8000/predict"  # Change this to your deployed model's URL

# Step 4: Send data in batches to the API and get predictions
# Since this is just an example, we'll process one sample at a time here.

predictions = []
start_time = time.time()
print(f"Sending {len(features)} samples to the model...")
for i, row in features.iterrows():
    sample = row.to_dict()  # Convert row to a dictionary
    print(f"Sending sample {i}: {sample}")
    response = requests.post(api_url, json=[sample])  # Send the request

    if response.status_code == 200:
        prediction = response.json()['predictions'][0]  # Extract prediction
        predictions.append(prediction)
        print(f"Prediction: {prediction}, Actual: {data['species'][i]}")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        predictions.append(None)

# Step 5: Calculate the latency (Time taken to process all requests)
latency = time.time() - start_time
print(f"Latency: {latency:.4f} seconds for {len(features)} samples.")

# Step 6: Evaluate accuracy (if you have the ground truth 'species')
# For the sake of this example, we are assuming the model is a classification model.
# You would compare the model's predictions with the actual 'species' values.

actual_labels = data['species']
accuracy = sum([1 if pred == actual else 0 for pred, actual in zip(predictions, actual_labels)]) / len(actual_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
