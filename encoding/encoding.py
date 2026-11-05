import pandas as pd
import os

# ==========================================
# STEP 1: Load Feature-Engineered Dataset
# ==========================================

input_file = "dataset/feature_engineered_automobile.csv"

df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)


# ==========================================
# STEP 2: Separate Target Variable
# ==========================================

X = df.drop("price", axis=1)
y = df["price"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# STEP 3: Identify Categorical Columns
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)


# ==========================================
# STEP 4: One-Hot Encoding
# ==========================================

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print("\nShape after encoding:", X_encoded.shape)


# ==========================================
# STEP 5: Combine Features and Target
# ==========================================

model_ready_df = pd.concat(
    [X_encoded, y],
    axis=1
)


# ==========================================
# STEP 6: Check Dataset
# ==========================================

print("\nFirst 5 rows of encoded dataset:")
print(model_ready_df.head())

print("\nData types:")
print(model_ready_df.dtypes)

print("\nMissing values:")
print(model_ready_df.isnull().sum().sum())


# ==========================================
# STEP 7: Save Model-Ready Dataset
# ==========================================

os.makedirs("dataset", exist_ok=True)

output_file = "dataset/model_ready.csv"

model_ready_df.to_csv(
    output_file,
    index=False
)

print("\nEncoding completed successfully!")
print("Saved file:", output_file)
print("Final dataset shape:", model_ready_df.shape)


# ==========================================
# STEP 8: Verification
# ==========================================

verified_df = pd.read_csv(output_file)

print("\nSaved file verification:")
print("Shape:", verified_df.shape)
print("Missing values:", verified_df.isnull().sum().sum())
print("Duplicates:", verified_df.duplicated().sum())
print("Target column present:", "price" in verified_df.columns)