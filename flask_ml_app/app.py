from flask import Flask, request, render_template
import pickle
import numpy as np

# ایجاد اپلیکیشن Flask
app = Flask(__name__)

# بارگذاری مدل
with open("model/model.pkl", "rb") as file:
    model = pickle.load(file)

# صفحه اصلی
@app.route("/")
def home():
    return render_template("index.html")

# نقطه پیش‌بینی
@app.route("/predict", methods=["POST"])
def predict():
    # دریافت داده‌ها از فرم
    features = [float(x) for x in request.form.values()]
    features_array = np.array(features).reshape(1, -1)

    # پیش‌بینی
    prediction = model.predict(features_array)[0]
    return render_template("index.html", prediction_text=f"نتیجه پیش‌بینی: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
