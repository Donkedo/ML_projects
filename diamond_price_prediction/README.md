# 💎 Diamond Price Prediction

A Machine Learning web application built with **Streamlit** that predicts the price of a diamond based on its physical characteristics. The model is trained using **XGBoost Regressor** and deployed with Streamlit for an interactive user experience.

---

## 📌 Project Overview

This application allows users to enter a diamond's characteristics, such as carat, dimensions, cut, color, and clarity, and instantly receive a predicted price.

The project demonstrates a complete end-to-end machine learning workflow, including:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training and evaluation
* Model serialization
* Streamlit web application
* Deployment

---

## 📂 Project Structure

```text
diamond_price_prediction/
│
├── app.py                  # Streamlit application
├── model.pkl               # Trained XGBoost model
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
```

---

## 🚀 Features

* Interactive web interface
* Real-time price prediction
* User-friendly input controls
* XGBoost regression model
* Fast and lightweight deployment

---

## 🧠 Input Features

### Numerical Features

* Carat
* Depth
* Table
* X Dimension
* Y Dimension
* Z Dimension

### Categorical Features

**Cut**

* Fair
* Good
* Very Good
* Premium
* Ideal

**Color**

* D(Best)
* E
* F
* G
* H
* I
* J(Wrost)

**Clarity**

* I1(Worst)
* SI2
* SI1
* VS2
* VS1
* VVS2
* VVS1
* IF(Best)

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

---

## ▶️ Running the Project Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd diamond_price_prediction
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open automatically in your default web browser.

---

## 📊 Model

* Algorithm: XGBoost Regressor
* Target Variable: Diamond Price
* Task: Regression

---

## 💡 Future Improvements

* Better UI styling
* Model explainability using SHAP
* Input validation
* Prediction confidence estimates
* Docker support
* Cloud deployment enhancements

---

## 👤 Author

**Aaman Manzar**

GitHub: https://github.com/donkedo
App   : https://mlprojects-aaman.streamlit.app/
