import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# INITIAL SETUP
# ==========================================

# Create Visuals Folder
os.makedirs("visuals", exist_ok=True)

# Load Dataset
df = pd.read_csv("dataset/cleaned_automobile.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
# STEP 2: DATASET UNDERSTANDING
# ==========================================

# 1. Dataset Information
print("\n========== DATASET INFORMATION ==========")
df.info()

# 2. Statistical Summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 3. Missing Values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 4. Duplicate Records
print("\n========== DUPLICATE RECORDS ==========")
print("Total Duplicates:", df.duplicated().sum())

# 5. Unique Values
print("\n========== UNIQUE VALUES ==========")
print(df.nunique())


# ==========================================
# STEP 3: UNIVARIATE ANALYSIS
# ==========================================

# Separate Numerical and Categorical Columns
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()


# 1. Histograms
df[numerical_cols].hist(
    figsize=(18, 15),
    bins=20,
    edgecolor="black"
)

plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.savefig("visuals/numerical_distributions.png", dpi=300)
plt.show()
plt.close()


# 2. Boxplots
plt.figure(figsize=(18, 10))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(5, 6, i)
    sns.boxplot(y=df[col])
    plt.title(col, fontsize=8)

plt.tight_layout()
plt.savefig("visuals/numerical_boxplots.png", dpi=300)
plt.show()
plt.close()


# 3. Countplots
for col in categorical_cols:
    plt.figure(figsize=(10, 5))

    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index
    )

    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"visuals/{col}_countplot.png", dpi=300)
    plt.show()
    plt.close()


# ==========================================
# STEP 4: BIVARIATE ANALYSIS
# ==========================================

# 1. Numerical Features vs Price

features = [
    "engine-size",
    "horsepower",
    "curb-weight",
    "city-mpg",
    "highway-mpg"
]

for col in features:
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x=col,
        y="price"
    )

    plt.title(f"{col} vs Car Price")
    plt.xlabel(col)
    plt.ylabel("Price")
    plt.tight_layout()

    plt.savefig(f"visuals/{col}_vs_price.png", dpi=300)
    plt.show()
    plt.close()


# 2. Categorical Features vs Price

categorical_features = [
    "make",
    "body-style",
    "fuel-type",
    "drive-wheels",
    "engine-type"
]

for col in categorical_features:
    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x=col,
        y="price"
    )

    plt.title(f"{col} vs Car Price")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"visuals/{col}_vs_price.png", dpi=300)
    plt.show()
    plt.close()


# 3. Correlation Heatmap

plt.figure(figsize=(18, 12))

correlation = df.select_dtypes(include=np.number).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Automobile Dataset")
plt.tight_layout()

plt.savefig("visuals/correlation_heatmap.png", dpi=300)
plt.show()
plt.close()


print("\n===================================")
print("EDA ANALYSIS COMPLETED SUCCESSFULLY")
print("All visualizations saved in visuals folder")
print("===================================")

# ==========================================
# STEP 5: PRICE CORRELATION ANALYSIS
# ==========================================

# Calculate correlation with target variable
price_correlation = correlation["price"].sort_values(
    ascending=False
)

print("\n========== PRICE CORRELATION ==========")
print(price_correlation)

# Plot correlation with price
plt.figure(figsize=(10, 8))

sns.barplot(
    x=price_correlation.values,
    y=price_correlation.index
)

plt.title("Feature Correlation with Car Price")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig("visuals/price_correlation.png", dpi=300)
plt.show()
plt.close()

# ==========================================
# STEP 6: OUTLIER ANALYSIS
# ==========================================

outlier_data = []

for col in numerical_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]

    outlier_data.append({
        "Feature": col,
        "Outliers": len(outliers)
    })

outlier_df = pd.DataFrame(outlier_data)

print("\n========== OUTLIER SUMMARY ==========")
print(outlier_df.to_string(index=False))

# Save outlier summary
outlier_df.to_csv(
    "visuals/outlier_summary.csv",
    index=False
)

# ==========================================
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# ==========================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Separate Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# Identify column types
num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = X.select_dtypes(include="object").columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features)
    ]
)

# Transform Features
X_processed = preprocessor.fit_transform(X)

# Train Random Forest for exploratory feature importance
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf.fit(X_processed, y)

# Get Feature Names
feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 15 IMPORTANT FEATURES ==========")
print(importance_df.head(15).to_string(index=False))

# Visualization
plt.figure(figsize=(12, 8))

sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Features Affecting Automobile Price Prediction")
plt.tight_layout()

plt.savefig("visuals/feature_importance.png", dpi=300)
plt.show()
plt.close()

# Save Feature Importance
importance_df.to_csv(
    "visuals/feature_importance.csv",
    index=False
)

# ==========================================
# STEP 8: EDA COMPLETION
# ==========================================

print("\n======================================")
print("EXPLORATORY DATA ANALYSIS COMPLETED")
print("======================================")

print("Total Records:", df.shape[0])
print("Total Features:", df.shape[1])

print("\nEDA Outputs:")
print("1. Numerical Distributions")
print("2. Numerical Boxplots")
print("3. Categorical Countplots")
print("4. Numerical Features vs Price")
print("5. Categorical Features vs Price")
print("6. Correlation Heatmap")
print("7. Price Correlation")
print("8. Outlier Summary")
print("9. Feature Importance")

print("\nAll visualizations saved in visuals folder.")