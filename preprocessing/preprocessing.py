
import pandas as pd

# Step 1:Load the original automobile dataset
df = pd.read_csv("automobile_uci.csv")

# Display the first 5 rows
print(df.head())

# Display dataset dimensions
print("Dataset Shape:", df.shape)


# Step 2: Check missing values

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

# Check '?' values in the dataset
print("\nQuestion mark values in each column:")
print((df.astype(str) == '?').sum())


# Step 3: Replace '?' with proper missing values

df = df.replace('?', pd.NA)

print("\nMissing values after replacement:")
print(df.isnull().sum())

print("\nTotal missing values after replacement:")
print(df.isnull().sum().sum())


# Step 4: Check data types

print("\nData types of all columns:")
print(df.dtypes)

print("\nDataset information:")
df.info()


# Step 5: Convert numerical columns to numeric data types

numeric_columns = [
    "symboling",
    "normalized-losses",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-size",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nUpdated data types:")
print(df[numeric_columns].dtypes)

print("\nMissing values after conversion:")
print(df.isnull().sum())


# Step 6: Remove rows with missing target values

print("Rows before removing missing prices:", len(df))

df = df.dropna(subset=["price"]).copy()

print("Rows after removing missing prices:", len(df))
print("Missing prices remaining:", df["price"].isnull().sum())

# Step 7: Check and remove duplicate rows

print("Duplicate rows before removal:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Duplicate rows after removal:", df.duplicated().sum())
print("Dataset shape after removing duplicates:", df.shape)

# Step 8: Fill missing numerical values using median

for col in numeric_columns:
    if col in df.columns and df[col].isnull().any():
        median_value = df[col].median()

        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)

print("Missing values in numerical columns:")
print(df[numeric_columns].isnull().sum())

# Step 9: Fill missing categorical values using mode

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for col in categorical_columns:
    if df[col].isnull().any():
        mode_values = df[col].mode(dropna=True)

        if not mode_values.empty:
            df[col] = df[col].fillna(mode_values.iloc[0])

print("Missing values in categorical columns:")
print(df[categorical_columns].isnull().sum())

print("\nTotal missing values remaining:")
print(df.isnull().sum().sum())

# Step 10: Save the new cleaned dataset

output_file = "cleaned_automobile.csv"

df.to_csv(output_file, index=False)

print("\nDataset saved successfully!")
print("File name:", output_file)
print("Final dataset shape:", df.shape)
print("Total missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Target column present:", "price" in df.columns)

# Verify the saved file
cleaned_df = pd.read_csv(output_file)

print("\nSaved file verification:")
print("Saved dataset shape:", cleaned_df.shape)
print("Saved file missing values:", cleaned_df.isnull().sum().sum())

print("\nFirst 5 rows of cleaned dataset:")
print(cleaned_df.head())
