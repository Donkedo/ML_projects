import streamlit as st
import joblib
import pandas as pd
import time

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("💎 Diamond Price Prediction")
st.write("Enter the diamond details below to predict its price.")


carat = st.number_input(
    "Carat",
    min_value=0.10,
    max_value=5.00,
    value=1.00,
    step=0.01
)

depth = st.slider(
    "Depth",
    min_value=40.0,
    max_value=80.0,
    value=61.5,
    step=0.1
)

table = st.slider(
    "Table",
    min_value=40,
    max_value=100,
    value=57
)

col1, col2, col3 = st.columns(3)

with col1:
    x = st.number_input(
        "Length (x)",
        min_value=0.01,
        max_value=15.0,
        value=5.00,
        step=0.01
    )

with col2:
    y = st.number_input(
        "Width (y)",
        min_value=0.01,
        max_value=15.0,
        value=5.00,
        step=0.01
    )

with col3:
    z = st.number_input(
        "Depth (z)",
        min_value=0.01,
        max_value=10.0,
        value=3.10,
        step=0.01
    )

cut = st.selectbox(
    "Cut",
    ["Fair", "Good", "Very Good", "Premium", "Ideal"]
)

color = st.selectbox(
    "Color",
    ["D", "E", "F", "G", "H", "I", "J"]
)

clarity = st.selectbox(
    "Clarity",
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
)

cut_Good = int(cut == "Good")
cut_Ideal = int(cut == "Ideal")
cut_Premium = int(cut == "Premium")
cut_Very_Good = int(cut == "Very Good")

color_E = int(color == "E")
color_F = int(color == "F")
color_G = int(color == "G")
color_H = int(color == "H")
color_I = int(color == "I")
color_J = int(color == "J")

clarity_IF = int(clarity == "IF")
clarity_SI1 = int(clarity == "SI1")
clarity_SI2 = int(clarity == "SI2")
clarity_VS1 = int(clarity == "VS1")
clarity_VS2 = int(clarity == "VS2")
clarity_VVS1 = int(clarity == "VVS1")
clarity_VVS2 = int(clarity == "VVS2")


feature_names = [
    "carat",
    "depth",
    "table",
    "x",
    "y",
    "z",
    "cut_Good",
    "cut_Ideal",
    "cut_Premium",
    "cut_Very Good",     
    "color_E",
    "color_F",
    "color_G",
    "color_H",
    "color_I",
    "color_J",
    "clarity_IF",
    "clarity_SI1",
    "clarity_SI2",
    "clarity_VS1",
    "clarity_VS2",
    "clarity_VVS1",
    "clarity_VVS2"
]

input_data = pd.DataFrame([[
    carat,
    depth,
    table,
    x,
    y,
    z,
    cut_Good,
    cut_Ideal,
    cut_Premium,
    cut_Very_Good,
    color_E,
    color_F,
    color_G,
    color_H,
    color_I,
    color_J,
    clarity_IF,
    clarity_SI1,
    clarity_SI2,
    clarity_VS1,
    clarity_VS2,
    clarity_VVS1,
    clarity_VVS2
]], columns=feature_names)

if st.button("Predict Price 💎"):

    if x <= 0 or y <= 0 or z <= 0:
        st.error("Diamond dimensions must be greater than zero.")
        st.stop()

    with st.spinner("Predicting price..."):
        time.sleep(1.5)

        try:
            prediction = model.predict(input_data)

            st.success(
                f"💰 Estimated Diamond Price: **${prediction[0]:,.2f} USD**"
            )

        except Exception as e:
            st.error(f"Prediction failed.\n\n{e}")
