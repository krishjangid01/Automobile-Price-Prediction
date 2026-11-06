# 🚗 Automobile Price Prediction Using Machine Learning

## 📌 Project Overview

This project aims to develop a Machine Learning system that predicts the price of automobiles based on various features such as engine size, horsepower, curb weight, fuel type, body style, mileage, and other vehicle specifications.

The project follows a complete machine learning workflow including data preprocessing, exploratory data analysis, feature engineering, categorical encoding, feature selection, model training, model evaluation, and a user-input-based automobile price prediction system.

---

## 🎯 Objectives

- Analyze automobile datasets to understand pricing patterns.
- Perform data cleaning and preprocessing.
- Identify important features affecting automobile prices.
- Perform feature engineering and categorical encoding.
- Select relevant features for model development.
- Build regression models to predict automobile prices.
- Evaluate and compare different machine learning algorithms.
- Select the best-performing model.
- Develop a user-input-based automobile price prediction system.
- Validate the final prediction system using real dataset records.

---

## 📂 Project Structure

```text
Automobile-Price-Prediction/
│
├── dataset/
│   ├── automobile_uci.csv
│   ├── cleaned_automobile.csv
│   ├── feature_engineered_automobile.csv
│   ├── model_ready.csv
│   ├── selected_features.csv
│   └── split/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── preprocessing/
│   └── preprocessing.py
│
├── feature_engineering/
│   └── feature_engineering.py
│
├── encoding/
│   └── encoding.py
│
├── feature_selection/
│   ├── feature_selection.py
│   ├── feature_price_correlation.csv
│   └── feature_price_correlation.png
│
├── eda/
│   └── eda.py
│
├── visuals/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── numerical_distributions.png
│   ├── numerical_boxplots.png
│   └── ...
│
├── model/
│   └── results/
│       ├── linear_regression_results.csv
│       ├── decision_tree_results.csv
│       ├── random_forest_results.csv
│       └── model_comparison.csv
│
├── models/
│   ├── linear_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── model_comparison.py
│   ├── train_test_split.py
│   ├── save_model.py
│   └── saved_models/
│       └── linear_regression_model.pkl
│
├── main.py
├── README.md
└── .gitignore
```

---

## 📊 Dataset Information

- **Dataset:** Automobile Dataset
- **Source:** UCI Machine Learning Repository
- **Original Records:** 205
- **Cleaned Records:** 201
- **Total Columns:** 26
- **Target Variable:** `price`

The dataset contains automobile specifications including:

- Make
- Fuel type
- Aspiration
- Number of doors
- Body style
- Drive wheels
- Engine location
- Wheel base
- Vehicle dimensions
- Curb weight
- Engine size
- Engine type
- Number of cylinders
- Fuel system
- Bore
- Stroke
- Compression ratio
- Horsepower
- Peak RPM
- City MPG
- Highway MPG

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- VS Code
- Git
- GitHub

---

# ⚙️ Project Workflow

## 1. Data Collection

The Automobile Dataset was collected from the UCI Machine Learning Repository.

The original dataset contains **205 automobile records and 26 columns**.

---

## 2. Data Preprocessing

The following preprocessing operations were performed:

- Handled missing values.
- Replaced missing value indicators (`?`) with appropriate missing values.
- Converted numerical columns to appropriate data types.
- Removed records with missing target values.
- Removed duplicate records.
- Filled missing numerical values using median values.
- Filled missing categorical values using mode values.
- Saved the cleaned dataset.

Final cleaned dataset:

```text
dataset/cleaned_automobile.csv
```

Dataset shape after preprocessing:

```text
201 rows × 26 columns
```

---

## 3. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset and identify relationships between automobile features and price.

The analysis included:

- Univariate Analysis
- Bivariate Analysis
- Correlation Analysis
- Outlier Detection
- Feature Importance Analysis
- Data Visualization

Several visualizations were generated and stored inside the `visuals/` directory.

---

## 4. Feature Engineering

Additional features were created to improve the representation of automobile characteristics.

The following engineered features were created:

- `average-mpg`
- `power-to-weight`
- `engine-weight-ratio`
- `vehicle-volume`

The feature-engineered dataset was saved as:

```text
dataset/feature_engineered_automobile.csv
```

---

## 5. Categorical Encoding

Categorical variables were converted into numerical form using **One-Hot Encoding**.

The encoded dataset was saved as:

```text
dataset/model_ready.csv
```

---

## 6. Feature Selection

Feature selection was performed using correlation analysis between the input features and the target variable `price`.

Features having an absolute correlation of at least `0.10` with the target variable were selected for model development.

The selected feature dataset was saved as:

```text
dataset/selected_features.csv
```

The final model uses **46 selected input features**.

---

## 7. Train-Test Split

The dataset was divided into training and testing sets using an 80:20 split.

- **Training Data:** 80%
- **Testing Data:** 20%
- **Random State:** 42

The resulting datasets were saved inside:

```text
dataset/split/
```

---

## 8. Model Development

Three regression algorithms were trained:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

---

## 9. Model Evaluation

The models were evaluated using the following regression metrics:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

## 10. Best Model Selection

The model comparison showed that **Linear Regression** performed the best on the test dataset based on all three evaluation metrics.

The trained model was saved using Joblib:

```text
models/saved_models/linear_regression_model.pkl
```

---

# 📈 Model Performance

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| **Linear Regression** | **1741.24** | **2563.18** | **0.9463** |
| Random Forest | 1921.30 | 3125.26 | 0.9202 |
| Decision Tree | 2146.51 | 3243.30 | 0.9140 |

### 🏆 Best Model: Linear Regression

The Linear Regression model achieved:

- **MAE:** 1741.24
- **RMSE:** 2563.18
- **R² Score:** 0.9463

An R² score of **0.9463** indicates that the model explains approximately **94.63% of the variation in automobile prices** on the test dataset.

---

# 📊 Key EDA Findings

- Engine size shows a strong positive correlation with automobile price.
- Curb weight and horsepower are strongly positively correlated with price.
- Vehicle width and length show positive relationships with price.
- City MPG and highway MPG show negative correlations with price.
- Engine size was identified as the most influential feature in the exploratory Random Forest feature importance analysis.
- Automobile price is strongly influenced by engine specifications, vehicle dimensions, and fuel efficiency.

---

# 🔮 Final Price Prediction System

The project includes a complete **user-input-based automobile price prediction system** implemented in `main.py`.

The system uses the trained Linear Regression model to predict automobile prices based on vehicle specifications entered by the user.

### The prediction system:

- Loads the trained model using Joblib.
- Accepts automobile specifications from the user.
- Performs the required feature engineering.
- Applies One-Hot Encoding to categorical inputs.
- Aligns the input data with the exact features used during model training.
- Generates the predicted automobile price.
- Displays the estimated price in a user-friendly format.

---

## 🧾 Prediction Inputs

The system accepts the following automobile specifications:

### Numerical Inputs

- Engine Size
- Curb Weight
- Horsepower
- Width
- Highway MPG
- City MPG
- Length
- Wheel Base
- Bore
- Height
- Peak RPM

### Categorical Inputs

- Make
- Drive Wheels
- Fuel System
- Number of Cylinders
- Engine Type
- Engine Location
- Body Style
- Aspiration
- Fuel Type

---

## ▶️ Run the Prediction System

After installing the required dependencies, run:

```bash
python3 main.py
```

The program will interactively ask for the automobile specifications and then display the predicted price.

Example output:

```text
==================================================
🚗 Automobile Price Prediction System
==================================================

Predicted Automobile Price: $13,567.05
```

---

# 🧪 Prediction System Validation

The final prediction system was validated using an actual automobile record from the cleaned dataset.

For one test automobile:

| Metric | Value |
|---|---:|
| Actual Price | $13,495.00 |
| Predicted Price | $14,217.35 |
| Absolute Difference | $722.35 |
| Percentage Error | ~5.35% |

The validation demonstrates that the complete pipeline—from user input and feature engineering to encoding, model prediction, and price output—is functioning successfully.

---

# 🚀 Installation & Setup

## Clone the Repository

```bash
git clone https://github.com/krishjangid01/Automobile-Price-Prediction.git
```

Navigate to the project directory:

```bash
cd Automobile-Price-Prediction
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## ▶️ Run Data Preprocessing

```bash
python3 preprocessing/preprocessing.py
```

## 📊 Run EDA

```bash
python3 eda/eda.py
```

## ⚙️ Run Feature Engineering

```bash
python3 feature_engineering/feature_engineering.py
```

## 🔢 Run Encoding

```bash
python3 encoding/encoding.py
```

## 🎯 Run Feature Selection

```bash
python3 feature_selection/feature_selection.py
```

## ✂️ Create Train-Test Split

```bash
python3 models/train_test_split.py
```

## 🤖 Train Models

```bash
python3 models/linear_regression.py
python3 models/decision_tree.py
python3 models/random_forest.py
```

## 📋 Compare Models

```bash
python3 models/model_comparison.py
```

## 💾 Save Best Model

```bash
python3 models/save_model.py
```

## 🔮 Run Price Prediction System

```bash
python3 main.py
```

---

# 📌 Project Status

## ✅ Completed

- [x] Data Collection
- [x] Data Preprocessing
- [x] Exploratory Data Analysis
- [x] Feature Engineering
- [x] Categorical Encoding
- [x] Feature Selection
- [x] Train-Test Split
- [x] Regression Model Development
- [x] Model Evaluation
- [x] Model Comparison
- [x] Best Model Selection
- [x] Trained Model Saving
- [x] User Input-Based Prediction System
- [x] Prediction Validation
- [x] Final Project Documentation

## 🎯 Final Result

The complete **Automobile Price Prediction System** has been successfully developed and tested.

The final system uses **Linear Regression** as the best-performing model and provides automobile price predictions based on user-provided vehicle specifications.

---

# 👨‍💻 Author

**Krish Jangid**

B.Tech Computer Science Engineering Student

GitHub: [@krishjangid01](https://github.com/krishjangid01)

---

⭐ If you find this project interesting, consider giving it a star!