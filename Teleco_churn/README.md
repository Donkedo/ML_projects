# 📊 Customer Churn Prediction

A Machine Learning web application built using **Streamlit** that predicts whether a customer is likely to churn based on demographic information, subscribed services, billing details, and contract information.

The application uses a trained **Logistic Regression** model and provides both the predicted class and the probability of churn.

---

# Project Workflow

The project follows a complete Machine Learning pipeline from data preprocessing to deployment.

## 1. Data Collection

* IBM Telco Customer Churn Dataset
* Customer demographic information
* Services subscribed
* Billing information
* Contract details

---

## 2. Data Preprocessing

The dataset was cleaned and prepared before training.

Steps performed:

* Removed unnecessary columns
* Converted categorical variables using One-Hot Encoding
* Converted target variable into binary values
* Standardized numerical features using **StandardScaler**
* Split the dataset into training and testing sets

---

## 3. Model Training

The processed dataset was used to train a **Logistic Regression** classifier.

Model pipeline:

* Data Preprocessing
* Feature Scaling
* Logistic Regression Training
* Model Evaluation
* Model Serialization using Joblib

---

## 4. Model Deployment

The trained model and scaler are saved using **Joblib**.

The Streamlit application performs the following steps:

1. Collects customer information from the user.
2. Converts categorical inputs into one-hot encoded features.
3. Arranges the features in the same order used during training.
4. Applies the saved StandardScaler.
5. Passes the scaled data to the trained Logistic Regression model.
6. Displays:

   * Churn Prediction
   * Churn Probability

---

# Libraries Used

The project is built using the following Python libraries:

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

# Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── logistic_regression.pkl
├── standard_scaler.pkl
├── notebook.ipynb
├── README.md
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<your-username>/Customer-Churn-Prediction.git
```

Move into the project directory.

```bash
cd Customer-Churn-Prediction
```

Install all required packages.

```bash
pip install -r requirements.txt
```

---

# Requirements

The project requires:

* Python 3.10 or later
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

# How to Run

After installing the dependencies, start the Streamlit application.

```bash
streamlit run app.py
```

The application will automatically open in your default web browser.

---

# Model Information

* **Algorithm:** Logistic Regression
* **Problem Type:** Binary Classification
* **Target Variable:** Churn

---

# Features Used

### Customer Information

* Gender
* Senior Citizen
* Partner
* Dependents

### Subscription Details

* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

### Billing Information

* Tenure
* Monthly Charges
* Total Charges
* Contract Type
* Paperless Billing
* Payment Method

---

# Prediction Output

The application predicts:

* Whether the customer is likely to churn.
* Probability of churn.

Example:

```text
Prediction:
Customer is likely to Stay

Churn Probability:
18.45%
```

---

# Future Improvements

* Add feature importance and explainability.
* Improve UI/UX.
* Compare multiple machine learning models.
* Deploy the application on Streamlit Community Cloud.
* Add downloadable prediction reports.

---

# Author

**Aaman Manzar**

B.Tech in Electronics and Telecommunications Engineering

Machine Learning | Data Science | Python
