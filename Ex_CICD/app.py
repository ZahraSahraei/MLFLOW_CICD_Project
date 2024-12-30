from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open(r"model.pkl", "rb") as input_file:
    model = pickle.load(input_file)

@app.route("/predict" , methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction" : prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0" , port=5000)

    