import pandas as pd
import os

# ==========================================
# STEP 1: Load Cleaned Dataset
# ==========================================

input_file = "dataset/cleaned_automobile.csv"

df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# STEP 2: Create Feature Engineering Folder
# ==========================================

os.makedirs("feature_engineering", exist_ok=True)


# ==========================================
# STEP 3: Create New Features
# ==========================================

# 1. Average MPG
df["average-mpg"] = (
    df["city-mpg"] + df["highway-mpg"]
) / 2


# 2. Power-to-weight ratio
df["power-to-weight"] = (
    df["horsepower"] / df["curb-weight"]
)


# 3. Engine-size to weight ratio
df["engine-weight-ratio"] = (
    df["engine-size"] / df["curb-weight"]
)


# 4. Vehicle volume
df["vehicle-volume"] = (
    df["length"] *
    df["width"] *
    df["height"]
)


# ==========================================
# STEP 4: Display New Features
# ==========================================

new_features = [
    "average-mpg",
    "power-to-weight",
    "engine-weight-ratio",
    "vehicle-volume"
]

print("\nNew Features Created:")
print(df[new_features].head())


# ==========================================
# STEP 5: Check Missing Values
# ==========================================

print("\nMissing values in new features:")
print(df[new_features].isnull().sum())


# ==========================================
# STEP 6: Check Dataset Information
# ==========================================

print("\nDataset shape after feature engineering:")
print(df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# ==========================================
# STEP 7: Save Feature-Engineered Dataset
# ==========================================

output_file = "dataset/feature_engineered_automobile.csv"

df.to_csv(output_file, index=False)

print("\nFeature engineering completed successfully!")
print("Saved file:", output_file)
print("Final dataset shape:", df.shape)


# ==========================================
# STEP 8: Verify Saved Dataset
# ==========================================

verified_df = pd.read_csv(output_file)

print("\nSaved file verification:")
print("Shape:", verified_df.shape)
print("Missing values:", verified_df.isnull().sum().sum())
print("Duplicates:", verified_df.duplicated().sum())
print("Target column present:", "price" in verified_df.columns)