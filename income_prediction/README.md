# 💰 Income Predictor

A Machine Learning web application built with **Python**, **Scikit-learn**, and **Streamlit** that predicts whether an individual's annual income is likely to be **greater than $50,000** based on demographic and employment-related information.

## 🚀 Live Demo

👉 https://mlprojects2-aaman.streamlit.app/

---

## 📌 Overview

This application uses a trained machine learning classification model to estimate whether a person's annual income exceeds **$50K**.

Users simply enter their information, and the application returns:

- ✅ Predicted income category
- 📊 Probability of earning more than $50K
- ⚡ Fast predictions through an interactive Streamlit interface

---

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Encoding Categorical Variables
5. Model Training
6. Model Evaluation
7. Model Serialization using Joblib
8. Streamlit Deployment

---

## 📂 Project Structure

```
Income-Predictor/
│
├── app.py
├── model.pkl
├── requirements.txt
├── notebook.ipynb
└── README.md
```

---

## 📊 Features Used

The model uses demographic and employment-related information including:

- Age
- Gender
- Work Class
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Region
- Capital Gain
- Capital Loss
- Hours Worked per Week

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📈 Model Output

The application predicts whether annual income is:

- **> $50K**
- **≤ $50K**

It also displays the probability associated with the prediction.

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/donkedo/income-predictor.git
```

Move into the project folder

```bash
cd income-predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

- Improve model accuracy through hyperparameter tuning
- Build preprocessing pipeline using Scikit-learn Pipeline
- Add feature importance visualization
- Improve UI/UX
- Docker deployment
- Cloud deployment with CI/CD

---

## 👨‍💻 Developer

**Aaman Manzar**

B.Tech Student | Machine Learning Enthusiast | Backend Development

GitHub:
https://github.com/donkedo

LinkedIn:
https://www.linkedin.com/in/aaman-manzar

---

## ⭐ If you found this project useful

Please consider giving the repository a ⭐ on GitHub.