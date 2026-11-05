import pandas as pd
import os
from sklearn.model_selection import train_test_split


# ==========================================
# STEP 1: Load Selected Feature Dataset
# ==========================================

input_file = "dataset/selected_features.csv"

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)


# ==========================================
# STEP 2: Separate Features and Target
# ==========================================

X = df.drop("price", axis=1)
y = df["price"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# STEP 3: Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# STEP 4: Display Split Information
# ==========================================

print("\nTrain-Test Split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ==========================================
# STEP 5: Save Split Datasets
# ==========================================

os.makedirs("dataset/split", exist_ok=True)

X_train.to_csv(
    "dataset/split/X_train.csv",
    index=False
)

X_test.to_csv(
    "dataset/split/X_test.csv",
    index=False
)

y_train.to_csv(
    "dataset/split/y_train.csv",
    index=False
)

y_test.to_csv(
    "dataset/split/y_test.csv",
    index=False
)


# ==========================================
# STEP 6: Verification
# ==========================================

print("\nTrain-Test Split completed successfully!")

print("\nFiles saved:")
print("dataset/split/X_train.csv")
print("dataset/split/X_test.csv")
print("dataset/split/y_train.csv")
print("dataset/split/y_test.csv")