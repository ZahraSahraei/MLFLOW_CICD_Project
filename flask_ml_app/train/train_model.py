from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# بارگذاری دیتاست
iris = load_iris()
X = iris.data
y = iris.target

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# آموزش مدل
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ذخیره مدل
with open("model/model.pkl", "wb") as file:
    pickle.dump(model, file)
    
