# 🚗 Automobile Price Prediction Using Machine Learning

## 📌 Project Overview

This project aims to develop a Machine Learning model that predicts the price of automobiles based on various features such as engine size, horsepower, curb weight, fuel type, body style, and other vehicle specifications.

The project uses Python and popular data science libraries to perform data preprocessing, exploratory data analysis, feature engineering, feature selection, and machine learning model development.

## 🎯 Objectives

- Analyze automobile datasets to understand pricing patterns.
- Perform data cleaning and preprocessing.
- Identify important features affecting automobile prices.
- Perform feature engineering and categorical encoding.
- Build regression models to predict automobile prices.
- Evaluate and compare different machine learning algorithms.
- Select the best-performing model for automobile price prediction.

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

## 📊 Dataset Information

- **Dataset:** Automobile Dataset
- **Source:** UCI Machine Learning Repository
- **Original Records:** 205
- **Cleaned Records:** 201
- **Total Columns:** 26
- **Target Variable:** `price`

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- VS Code
- Git & GitHub

## ⚙️ Project Workflow

### 1. Data Collection

Collected the Automobile Dataset containing vehicle specifications and automobile prices.

### 2. Data Preprocessing

- Handled missing values.
- Replaced missing value indicators (`?`) with appropriate missing values.
- Converted numerical columns to appropriate data types.
- Removed records with missing target values.
- Removed duplicate records.
- Filled missing numerical values using median values.
- Filled missing categorical values using mode values.
- Saved the cleaned dataset.

### 3. Exploratory Data Analysis (EDA)

Performed:

- Univariate Analysis
- Bivariate Analysis
- Correlation Analysis
- Outlier Detection
- Feature Importance Analysis
- Data Visualization

### 4. Feature Engineering

Created additional features to improve the representation of automobile characteristics:

- `average-mpg`
- `power-to-weight`
- `engine-weight-ratio`
- `vehicle-volume`

The feature-engineered dataset was saved as:

`dataset/feature_engineered_automobile.csv`

### 5. Categorical Encoding

Categorical variables were converted into numerical form using One-Hot Encoding.

The encoded dataset was saved as:

`dataset/model_ready.csv`

### 6. Feature Selection

Feature selection was performed using correlation analysis between input features and the target variable `price`.

Features with an absolute correlation of at least `0.10` were selected for model development.

The selected feature dataset was saved as:

`dataset/selected_features.csv`

### 7. Train-Test Split

The dataset was divided into training and testing sets using:

- **Training Data:** 80%
- **Testing Data:** 20%
- **Random State:** 42

The split datasets were saved inside:

`dataset/split/`

### 8. Model Development

Three regression models were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### 9. Model Evaluation

The models were evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

### 10. Best Model Selection

The model comparison showed that **Linear Regression** performed the best on the test dataset based on all three evaluation metrics.

The trained Linear Regression model was saved using Joblib at:

`models/saved_models/linear_regression_model.pkl`

## 📈 Model Performance

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

## 📊 Key EDA Findings

- Engine size shows a strong positive correlation with automobile price.
- Curb weight and horsepower are also strongly positively correlated with price.
- Vehicle width and length show positive relationships with price.
- City MPG and highway MPG show negative correlations with price.
- Engine size was identified as the most influential feature in the exploratory Random Forest feature importance analysis.
- Automobile price is strongly influenced by engine specifications, vehicle dimensions, and fuel efficiency.

## 🚀 Installation & Setup

Clone the repository:

```bash
git clone https://github.com/krishjangid01/Automobile-Price-Prediction.git
```

Navigate to the project directory:

```bash
cd Automobile-Price-Prediction
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Run Data Preprocessing

```bash
python3 preprocessing/preprocessing.py
```

### Run EDA

```bash
python3 eda/eda.py
```

### Run Feature Engineering

```bash
python3 feature_engineering/feature_engineering.py
```

### Run Encoding

```bash
python3 encoding/encoding.py
```

### Run Feature Selection

```bash
python3 feature_selection/feature_selection.py
```

### Create Train-Test Split

```bash
python3 models/train_test_split.py
```

### Train Models

```bash
python3 models/linear_regression.py
python3 models/decision_tree.py
python3 models/random_forest.py
```

### Compare Models

```bash
python3 models/model_comparison.py
```

### Save Best Model

```bash
python3 models/save_model.py
```

## 📌 Project Status

### ✅ Completed

- Data Collection
- Data Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Categorical Encoding
- Feature Selection
- Train-Test Split
- Regression Model Development
- Model Evaluation
- Model Comparison
- Best Model Selection
- Trained Model Saving

### 🔄 Next Step

- Develop the final automobile price prediction system.
- Integrate the saved model with `main.py`.
- Add user input-based price prediction.
- Perform final testing and validation.
- Update the final project documentation.

## 👨‍💻 Author

**Krish Jangid**

B.Tech Computer Science Engineering Student

GitHub: [@krishjangid01](https://github.com/krishjangid01)

---

⭐ If you find this project interesting, consider giving it a star!