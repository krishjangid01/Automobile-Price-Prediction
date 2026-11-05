import pandas as pd
import os
import matplotlib.pyplot as plt


# ==========================================
# STEP 1: Load Model-Ready Dataset
# ==========================================

input_file = "dataset/model_ready.csv"

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
# STEP 3: Calculate Correlation with Price
# ==========================================

correlations = df.corr(numeric_only=True)["price"].drop("price")

correlations = correlations.sort_values(
    key=abs,
    ascending=False
)

print("\nFeature correlation with price:")
print(correlations)


# ==========================================
# STEP 4: Save Correlation Results
# ==========================================

os.makedirs("feature_selection", exist_ok=True)

correlation_file = "feature_selection/feature_price_correlation.csv"

correlations.to_csv(
    correlation_file,
    header=["correlation"]
)

print("\nCorrelation results saved to:")
print(correlation_file)


# ==========================================
# STEP 5: Select Important Features
# ==========================================

threshold = 0.10

selected_features = correlations[
    correlations.abs() >= threshold
].index.tolist()

print("\nSelected features:")
print(selected_features)

print("\nNumber of selected features:", len(selected_features))


# ==========================================
# STEP 6: Create Selected Feature Dataset
# ==========================================

X_selected = X[selected_features]

selected_df = pd.concat(
    [X_selected, y],
    axis=1
)

output_file = "dataset/selected_features.csv"

selected_df.to_csv(
    output_file,
    index=False
)

print("\nSelected feature dataset saved:")
print(output_file)

print("\nFinal selected dataset shape:")
print(selected_df.shape)


# ==========================================
# STEP 7: Create Correlation Plot
# ==========================================

plt.figure(figsize=(10, 8))

correlations.sort_values().plot(
    kind="barh"
)

plt.title("Feature Correlation with Automobile Price")
plt.xlabel("Correlation")
plt.ylabel("Features")
plt.tight_layout()

plot_file = "feature_selection/feature_price_correlation.png"

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nCorrelation plot saved:")
print(plot_file)


# ==========================================
# STEP 8: Final Verification
# ==========================================

print("\nFinal verification:")
print("Missing values:", selected_df.isnull().sum().sum())
print("Duplicates:", selected_df.duplicated().sum())
print("Target column present:", "price" in selected_df.columns)

print("\nFeature selection completed successfully!")