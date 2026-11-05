import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# STEP 1: Load Training and Testing Data
# ==========================================

X_train = pd.read_csv("dataset/split/X_train.csv")
X_test = pd.read_csv("dataset/split/X_test.csv")

y_train = pd.read_csv("dataset/split/y_train.csv").squeeze()
y_test = pd.read_csv("dataset/split/y_test.csv").squeeze()


# ==========================================
# STEP 2: Create Random Forest Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)


# ==========================================
# STEP 3: Train Model
# ==========================================

model.fit(X_train, y_train)

print("Random Forest model trained successfully!")


# ==========================================
# STEP 4: Make Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# STEP 5: Evaluate Model
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\nRandom Forest Results:")
print("--------------------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)


# ==========================================
# STEP 6: Save Results
# ==========================================

os.makedirs("model/results", exist_ok=True)

results = pd.DataFrame({
    "Model": ["Random Forest"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

results.to_csv(
    "model/results/random_forest_results.csv",
    index=False
)

print("\nResults saved successfully!")
print("File: model/results/random_forest_results.csv")