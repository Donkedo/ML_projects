# 💎 Diamond Price Prediction

A machine learning web application built with **Streamlit** that predicts diamond prices based on their physical characteristics. The application uses an **XGBoost Regressor** to provide interactive price predictions.

🚀 **Live Demo:** [Try the Diamond Price Prediction App](https://mlprojects-aaman.streamlit.app/)

💻 **GitHub Repository:** [Diamond Price Prediction](https://github.com/Donkedo/ML_projects/tree/main/diamond_price_prediction)

---

## 📌 Project Overview

This application allows users to enter a diamond's characteristics, such as carat, dimensions, cut, color, and clarity, and receive a predicted price.

The project demonstrates an end-to-end machine learning workflow, including:

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering and encoding
- Model training and evaluation
- Model serialization
- Streamlit application development
- Deployment

---

## 📂 Project Structure

```text
diamond_price_prediction/
│
├── app.py                  # Streamlit application
├── model.pkl               # Trained XGBoost model
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
└── diamond-price-prediction.ipynb  # Model development and EDA
```

---

## 🚀 Features

- Interactive web interface
- Real-time diamond price predictions
- User-friendly input controls
- XGBoost regression model
- Data-driven price estimation
- Web-based deployment using Streamlit

---

## 🧠 Input Features

### Numerical Features

- Carat
- Depth
- Table
- X Dimension
- Y Dimension
- Z Dimension

### Categorical Features

**Cut**
- Fair
- Good
- Very Good
- Premium
- Ideal

**Color**
- D (Best)
- E
- F
- G
- H
- I
- J (Lowest grade)

**Clarity**
- I1 (Lowest grade)
- SI2
- SI1
- VS2
- VS1
- VVS2
- VVS1
- IF (Highest grade)

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Jupyter Notebook

---

## 🔄 Machine Learning Workflow

1. **Data Inspection:** Examined dataset structure, feature types, and descriptive statistics.
2. **Data Cleaning:** Removed the unnecessary index column and duplicate records.
3. **Exploratory Data Analysis:** Analyzed feature distributions, correlations, and relationships between diamond characteristics and prices.
4. **Feature Engineering:** Separated input features from the target variable and encoded categorical features using one-hot encoding.
5. **Data Splitting:** Divided the data into training and testing sets.
6. **Model Training:** Trained and compared Linear Regression, Ridge Regression, Lasso Regression, Decision Tree, Random Forest, and XGBoost models.
7. **Model Evaluation:** Compared the models using R², MAE, and RMSE.
8. **Model Selection:** Selected XGBoost based on its test-set R² and RMSE.
9. **Model Serialization:** Saved the trained model for use in the application.
10. **Deployment:** Built and deployed an interactive Streamlit application.

---

## 📊 Model Performance

The following metrics are from the recorded notebook results.

| Model | Test R² | MAE (USD) | RMSE (USD) |
|---|---:|---:|---:|
| Linear Regression | 0.9195 | 735.18 | 1,114.54 |
| Ridge Regression | 0.9194 | 735.87 | 1,114.79 |
| Lasso Regression | 0.9194 | 734.78 | 1,115.09 |
| Decision Tree | 0.9556 | 379.40 | 827.67 |
| Random Forest | 0.9738 | **294.76** | 635.69 |
| **XGBoost** | **0.9765** | 305.69 | **601.70** |

### Best Model: XGBoost Regressor

- **Test R²:** 0.9765
- **Test MAE:** $305.69
- **Test RMSE:** $601.70

XGBoost achieved the highest test R² and the lowest test RMSE among the evaluated models. Random Forest achieved a slightly lower MAE, indicating a smaller average absolute prediction error.

*Note: These are previously recorded experimental results. Reproduce the evaluation before treating them as final benchmark results.*

---

## ▶️ Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Donkedo/ML_projects.git
cd ML_projects/diamond_price_prediction
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

---

## 💡 Future Improvements

- Improve UI design and responsiveness.
- Integrate SHAP for model explainability.
- Add stronger input validation.
- Investigate prediction uncertainty.
- Containerize the application using Docker.
- Improve deployment and reproducibility.

---

## 👤 Author

**Aaman Manzar**

B.Tech in Electronics and Telecommunication Engineering  
Indian Institute of Engineering Science and Technology (IIEST), Shibpur

- **GitHub:** [@Donkedo](https://github.com/Donkedo)
- **Live Application:** [Diamond Price Prediction](https://mlprojects-aaman.streamlit.app/)

---

If you find this project useful, feel free to explore the repository and try the live application!
