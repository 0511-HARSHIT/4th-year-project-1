import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("energy_dataset.csv")

X = data[['temperature','humidity','occupancy','hour']]
y = data['energy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50),
    "Decision Tree": DecisionTreeRegressor(),
    "SVM": SVR()
}

results = {}
best_model = None
best_error = float("inf")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error = mean_absolute_error(y_test, pred)
    results[name] = round(error, 2)

    if error < best_error:
        best_error = error
        best_model = model

joblib.dump(best_model, "model.pkl")
joblib.dump(results, "results.pkl")

print("✅ Training Complete")