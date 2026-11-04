# 🚗 Automobile Price Prediction Using Machine Learning

## 📌 Project Overview

This project aims to develop a Machine Learning model that predicts the price of automobiles based on various features such as engine size, horsepower, curb weight, fuel type, body style, and other vehicle specifications.

The project uses Python and popular data science libraries to perform data preprocessing, exploratory data analysis, and machine learning model development.

## 🎯 Objectives

- Analyze automobile datasets to understand pricing patterns.
- Perform data cleaning and preprocessing.
- Identify important features affecting automobile prices.
- Build regression models to predict automobile prices.
- Evaluate and compare different machine learning algorithms.

## 📂 Project Structure

```text
Automobile-Price-Prediction/
│
├── dataset/
│   ├── automobile_uci.csv
│   └── cleaned_automobile.csv
│
├── preprocessing/
│   └── preprocessing.py
│
├── eda/
│   └── eda.py
│
├── visuals/
│   ├── numerical_distributions.png
│   ├── numerical_boxplots.png
│   ├── correlation_heatmap.png
│   ├── price_correlation.png
│   ├── feature_importance.png
│   └── ...
│
├── models/
│
├── main.py
├── README.md
└── .gitignore
```

## 📊 Dataset Information

- Dataset: Automobile Dataset
- Source: UCI Machine Learning Repository
- Original Records: 205
- Cleaned Records: 201
- Total Columns: 26
- Target Variable: `price`

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code
- Git & GitHub

## ⚙️ Project Workflow

### 1. Data Collection
Collected the automobile dataset containing vehicle specifications and prices.

### 2. Data Preprocessing
- Handled missing values.
- Converted appropriate data types.
- Removed records with missing target values.
- Performed data cleaning.
- Saved the cleaned dataset.

### 3. Exploratory Data Analysis (EDA)
- Univariate Analysis
- Bivariate Analysis
- Correlation Analysis
- Outlier Detection
- Feature Importance Analysis
- Data Visualization

### 4. Feature Engineering
Coming Soon...

### 5. Model Development
Coming Soon...

### 6. Model Evaluation
Coming Soon...

## 📈 Key EDA Findings

- Engine size shows a strong positive correlation with automobile price.
- Curb weight and horsepower are also strongly positively correlated with price.
- City MPG and highway MPG show negative correlations with price.
- Engine size was identified as the most important feature in the exploratory Random Forest analysis.

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
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the EDA script:

```bash
python3 eda/eda.py
```

## 📌 Project Status

Currently completed:
- Data Collection
- Data Preprocessing
- Exploratory Data Analysis

Upcoming:
- Feature Engineering
- Regression Model Development
- Model Evaluation
- Final Price Prediction System

## 👨‍💻 Author

**Krish Jangid**

B.Tech Computer Science Engineering Student

GitHub: [@krishjangid01](https://github.com/krishjangid01)

---

⭐ If you find this project interesting, consider giving it a star!