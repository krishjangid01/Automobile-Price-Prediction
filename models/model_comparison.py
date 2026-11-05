import pandas as pd
import os


# ==========================================
# STEP 1: Load Model Results
# ==========================================

linear = pd.read_csv(
    "model/results/linear_regression_results.csv"
)

decision_tree = pd.read_csv(
    "model/results/decision_tree_results.csv"
)

random_forest = pd.read_csv(
    "model/results/random_forest_results.csv"
)


# ==========================================
# STEP 2: Combine Results
# ==========================================

comparison = pd.concat(
    [linear, decision_tree, random_forest],
    ignore_index=True
)


# ==========================================
# STEP 3: Display Model Comparison
# ==========================================

print("\nModel Comparison:")
print("--------------------------------")

print(
    comparison.to_string(index=False)
)


# ==========================================
# STEP 4: Identify Best Models
# ==========================================

best_mae = comparison.loc[
    comparison["MAE"].idxmin()
]

best_rmse = comparison.loc[
    comparison["RMSE"].idxmin()
]

best_r2 = comparison.loc[
    comparison["R2"].idxmax()
]


print("\nBest Model based on MAE:")
print(best_mae["Model"])

print("\nBest Model based on RMSE:")
print(best_rmse["Model"])

print("\nBest Model based on R²:")
print(best_r2["Model"])


# ==========================================
# STEP 5: Save Comparison
# ==========================================

output_file = "model/results/model_comparison.csv"

comparison.to_csv(
    output_file,
    index=False
)

print("\nModel comparison saved successfully!")
print("File:", output_file)