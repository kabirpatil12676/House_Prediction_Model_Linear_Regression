---

# 🏠 House Price Prediction using Machine Learning

A supervised machine learning project that predicts residential house prices based on property features such as location, structure, and condition. This project demonstrates a complete **end-to-end regression pipeline** including data preprocessing, feature engineering, model training, and evaluation.

---

## 📌 Project Overview

Real estate pricing depends on multiple factors like area, zoning, building type, and basement size. This project builds a **Linear Regression model** to predict house prices using historical data.

The goal is to:

* Clean and preprocess real-world housing data
* Handle missing values and categorical variables
* Train and evaluate a regression model
* Measure performance using **R²** and **Adjusted R²**

---

## 🧠 Machine Learning Approach

* **Problem Type:** Supervised Learning – Regression
* **Algorithm Used:** Linear Regression
* **Target Variable:** `SalePrice`

---

## 🗂️ Dataset Description

The dataset (`HousePricePrediction.csv`) contains:

* Numerical features (LotArea, Basement size, etc.)
* Categorical features (Zoning, Building Type, Exterior, etc.)
* Missing values (handled during preprocessing)

---

## ⚙️ Data Preprocessing & Feature Engineering

### 1️⃣ Feature Selection

* Dropped irrelevant columns:

  * `Id`
  * `SalePrice` (used as target)

### 2️⃣ Handling Missing Values

* Target variable (`SalePrice`): filled with **mean**
* Basement-related features: filled with **0**
* Remaining numerical features: filled with **median**

### 3️⃣ Encoding Categorical Variables

* One-hot encoding using `pd.get_dummies()`
* Applied `drop_first=True` to avoid the dummy variable trap

### 4️⃣ Train-Test Split

* 80% Training
* 20% Testing
* `random_state = 42` for reproducibility

---

## 🧪 Model Training

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

---

## 📈 Model Evaluation

### Metrics Used:

* **R² Score**
* **Adjusted R² Score**

```python
r2 = r2_score(y_test, y_pred)

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
```

### Why Adjusted R²?

Adjusted R² accounts for the number of predictors and helps detect overfitting, making it more reliable for real-world regression problems.

---

## ✅ Results

* The model successfully captures the relationship between house features and sale price.
* Adjusted R² confirms that added features contribute meaningfully to predictions.

*(Exact values may vary depending on dataset version)*

---

## 🛠️ Tech Stack

* **Python**
* **Pandas** – Data manipulation
* **NumPy** – Numerical operations
* **Scikit-learn** – ML modeling & evaluation

---

## 🚀 How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   ```
2. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Run the script or notebook:

   ```bash
   python house_price_prediction.py
   ```

---

## 📌 Key Learnings

* Practical handling of missing values
* Encoding categorical data for ML models
* Understanding regression evaluation metrics
* Building a clean ML pipeline

---

## 🔮 Future Improvements

* Try advanced models (Ridge, Lasso, Random Forest)
* Perform feature scaling
* Add cross-validation
* Deploy as a web app using Flask or Streamlit

---

## 👤 Author

**Kabir Patil**
Machine Learning & Full-Stack Enthusiast
📌 Open to internships and ML-focused roles

---

