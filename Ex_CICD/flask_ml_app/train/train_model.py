import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# بارگذاری دیتاست از لینک
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# ویژگی‌ها (X) و هدف (y)
X = data.drop(columns=["medv"])  # medv ستون قیمت خانه است
y = data["medv"]

# تقسیم داده‌ها به داده‌های آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# آموزش مدل
model = RandomForestRegressor()
model.fit(X_train, y_train)

# ایجاد پوشه model (اگر وجود ندارد)
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# ذخیره مدل در فایل model.pkl
model_path = os.path.join(model_dir, "model.pkl")
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved to {model_path}")
