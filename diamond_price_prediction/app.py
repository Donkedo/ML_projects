import streamlit as st
import joblib 
import time
import pandas as pd
import numpy as np
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)
st.title("We are going to predict price of Diamonds")

carat = st.number_input(
    "Carat",
    min_value=0.1,
    max_value=5.0,
    value=1.0
)
depth = st.slider(
    "Depth",
    40.0,
    80.0,
    61.5
)
table=st.slider("Table",
                40,100,65)

col1,col2,col3=st.columns(3)
with col1:
    x = st.number_input(
    "X dimension (mm)",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.01
)
with col2:
   y = st.number_input(
    "Y dimension (mm)",
    min_value=0.0,
    max_value=60.0,
    value=5.0,
    step=0.01
)
with col3:
    z = st.number_input(
    "Z dimension (mm)",
    min_value=0.0,
    max_value=32.0,
    value=5.0,
    step=0.01
)

cut = st.selectbox(
    "Cut",
    ["Fair", "Good", "Very Good", "Premium", "Ideal"]
)
color=st.selectbox(
    "Color",
    ['J','I','H','G','F','E','D']
)
clarity = st.selectbox(
    "Clarity",
    ['I1' , 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
)


cut_Good = 0
cut_Ideal = 0
cut_Premium = 0
cut_Very_Good = 0

if cut == "Good":
    cut_Good = 1
elif cut == "Ideal":
    cut_Ideal = 1
elif cut == "Premium":
    cut_Premium = 1
elif cut == "Very Good":
    cut_Very_Good = 1


color_E = 0
color_F = 0
color_G = 0
color_H = 0
color_I = 0
color_J = 0

if color == "E":
    color_E = 1
elif color == "F":
    color_F = 1
elif color == "G":
    color_G = 1
elif color == "H":
    color_H = 1
elif color == "I":
    color_I = 1
elif color == "J":
    color_J = 1


clarity_IF = 0
clarity_SI1 = 0
clarity_SI2 = 0
clarity_VS1 = 0
clarity_VS2 = 0
clarity_VVS1 = 0
clarity_VVS2 = 0

if clarity == "IF":
    clarity_IF = 1
elif clarity == "SI1":
    clarity_SI1 = 1
elif clarity == "SI2":
    clarity_SI2 = 1
elif clarity == "VS1":
    clarity_VS1 = 1
elif clarity == "VS2":
    clarity_VS2 = 1
elif clarity == "VVS1":
    clarity_VVS1 = 1
elif clarity == "VVS2":
    clarity_VVS2 = 1


feature_names = [
    'carat','depth','table','x','y','z',
    'cut_Good','cut_Ideal','cut_Premium','cut_Very Good',
    'color_E','color_F','color_G','color_H','color_I','color_J',
    'clarity_IF','clarity_SI1','clarity_SI2',
    'clarity_VS1','clarity_VS2','clarity_VVS1','clarity_VVS2'
]

input_data = pd.DataFrame([[
    carat, depth, table, x, y, z,
    cut_Good, cut_Ideal, cut_Premium, cut_Very_Good,
    color_E, color_F, color_G, color_H, color_I, color_J,
    clarity_IF, clarity_SI1, clarity_SI2,
    clarity_VS1, clarity_VS2, clarity_VVS1, clarity_VVS2
]], columns=feature_names)

if st.button("predict price"):
    st.write("predicting..........")
    time.sleep(1)
    st.write("Asking miners.........")
    time.sleep(1)
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: {prediction[0]:,.2f} USD")
