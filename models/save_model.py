import pandas as pd
import joblib
import os

from sklearn.linear_model import LinearRegression


# ==========================================
# STEP 1: Load Training Data
# ==========================================

X_train = pd.read_csv("dataset/split/X_train.csv")
y_train = pd.read_csv("dataset/split/y_train.csv").squeeze()

print("Training data loaded successfully!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# ==========================================
# STEP 2: Train Best Model
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")


# ==========================================
# STEP 3: Create Model Directory
# ==========================================

os.makedirs("models/saved_models", exist_ok=True)


# ==========================================
# STEP 4: Save Model
# ==========================================

model_path = "models/saved_models/linear_regression_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Model path:", model_path)


# ==========================================
# STEP 5: Verify Saved Model
# ==========================================

loaded_model = joblib.load(model_path)

print("\nSaved model verification:")
print("Model type:", type(loaded_model).__name__)
print("Number of coefficients:", len(loaded_model.coef_))

print("\nBest model is ready for prediction! 🚀")