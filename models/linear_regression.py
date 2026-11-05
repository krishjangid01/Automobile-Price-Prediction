import pandas as pd
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# STEP 1: Load Training and Testing Data
# ==========================================

X_train = pd.read_csv("dataset/split/X_train.csv")
X_test = pd.read_csv("dataset/split/X_test.csv")

y_train = pd.read_csv("dataset/split/y_train.csv").squeeze()
y_test = pd.read_csv("dataset/split/y_test.csv").squeeze()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ==========================================
# STEP 2: Create Linear Regression Model
# ==========================================

model = LinearRegression()


# ==========================================
# STEP 3: Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")


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


print("\nLinear Regression Results:")
print("--------------------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)


# ==========================================
# STEP 6: Save Results
# ==========================================

os.makedirs("model/results", exist_ok=True)

results = pd.DataFrame({
    "Model": ["Linear Regression"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

results.to_csv(
    "model/results/linear_regression_results.csv",
    index=False
)

print("\nResults saved successfully!")
print("File: model/results/linear_regression_results.csv")