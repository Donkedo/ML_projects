from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import time

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Income predictor",
    page_icon="$",
    layout="centered"
)
st.title("Income predictor")
st.write("This application predicts whether an individual's annual income is likely to exceed $50,000 using demographic and employment-related features. The model was trained on the Adult Census Income dataset using machine learning.")
age=st.slider("Age",10,100,25,1)

sex=st.selectbox("Sex",['Male','Female'])  #male was 1 and female was 0
sex = 1 if sex=="Male" else 0
col1,col2=st.columns(2)
with col1:
    capital_gain=st.number_input("Capital Gain ($)",0,100000,0,10)
with col2:
    capital_loss=st.number_input("Capital Loss ($)",0,5000,0,1)
work_hours=st.number_input("Hours Worked per Week",0,100,40,1)

work_class=st.selectbox(
    "You work class",
    [
    'Private',
    'State government',
    'Federal government',
    'Self employed not in cooperative',
    'Self employed in cooperative',
    'Local government',
    'Without-pay'
    ])


education = st.selectbox(
    "Education",
    [
        "Preschool",
        "1st-4th",
        "5th-6th",
        "7th-8th",
        "9th",
        "10th",
        "11th",
        "12th",
        "HS-grad",
        "Some-college",
        "Associate Degree (Vocational/Technical)",
        "Associate Degree (Academic)",
        "Bachelors",
        "Masters",
        "Prof-school",
        "Doctorate"
    ]
)

marital_status=st.selectbox(
    'Marital Status'
    ,[
        'married',
        'not-married',
        'divorced',
        'seperated',
        'widowed',
        'married(spouse live elsewhere)',
        'married(spouse is in armed force)'
        ])

occupation=st.selectbox(
    'Occupation'
    ,[
        'Executive-managerial',
        'Machine-operator-inspector', 
        'Prof-specialty',
       'Other-service',
        'Adm-clerical', 
        'Transport-moving', 
        'Sales',
       'Craft-repair', 
       'Farming-fishing', 
       'Tech-support',
       'Protective-service',
        'Handlers-cleaners', 
        'Armed-Forces',
       'Private-house-service'
       ])

relationship=st.selectbox(
    "Relationship",
    [
    'Not in family', 
    'Unmarried', 
    'Own child', 
    'Other relative',
    'Husband', 
    'Wife'
    ])

race=st.selectbox(
    "Race",
    [
        'black',
        'white',
        'other'
    ]
)

region=st.selectbox(
    "Region of work",
    [
    'North America', 
    'Europe',
    'Asia',
    'Central America & Caribbean',
    'Africa',
    'South America',
    'Oceania'
    ]
)
features=[
    'age','sex',
 'capital.gain',
 'capital.loss',
 'hours.per.week',
 'workclass_Local-gov',
 'workclass_Private',
 'workclass_Self-emp-inc',
 'workclass_Self-emp-not-inc',
 'workclass_State-gov',
 'workclass_Without-pay',
 'education_11th',
 'education_12th',
 'education_1st-4th',
 'education_5th-6th',
 'education_7th-8th',
 'education_9th',
 'education_Assoc-acdm',
 'education_Assoc-voc',
 'education_Bachelors',
 'education_Doctorate',
 'education_HS-grad',
 'education_Masters',
 'education_Preschool',
 'education_Prof-school',
 'education_Some-college',
 'marital.status_Married-AF-spouse',
 'marital.status_Married-civ-spouse',
 'marital.status_Married-spouse-absent',
 'marital.status_Never-married',
 'marital.status_Separated',
 'marital.status_Widowed',
 'occupation_Armed-Forces',
 'occupation_Craft-repair',
 'occupation_Exec-managerial',
 'occupation_Farming-fishing',
 'occupation_Handlers-cleaners',
 'occupation_Machine-op-inspct',
 'occupation_Other-service',
 'occupation_Priv-house-serv',
 'occupation_Prof-specialty',
 'occupation_Protective-serv',
 'occupation_Sales',
 'occupation_Tech-support',
 'occupation_Transport-moving',
 'relationship_Not-in-family',
 'relationship_Other-relative',
 'relationship_Own-child',
 'relationship_Unmarried',
 'relationship_Wife',
 'race_Other',
 'race_White',
 'region_Asia',
 'region_Central America & Caribbean',
 'region_Europe',
 'region_North America',
 'region_Oceania',
 'region_South America']


feature_values = {feature: 0 for feature in features}

feature_values["age"] = age
feature_values["sex"] = sex
feature_values["capital.gain"] = capital_gain
feature_values["capital.loss"] = capital_loss
feature_values["hours.per.week"] = work_hours

feature_values["workclass_Local-gov"] = int(work_class == "Local government")
feature_values["workclass_Private"] = int(work_class == "Private")
feature_values["workclass_Self-emp-inc"] = int(work_class == "Self employed in cooperative")
feature_values["workclass_Self-emp-not-inc"] = int(work_class == "Self employed not in cooperative")
feature_values["workclass_State-gov"] = int(work_class == "State government")
feature_values["workclass_Without-pay"] = int(work_class == "Without-pay")



feature_values["education_11th"] = int(education == "11th")
feature_values["education_12th"] = int(education == "12th")
feature_values["education_1st-4th"] = int(education == "1st-4th")
feature_values["education_5th-6th"] = int(education == "5th-6th")
feature_values["education_7th-8th"] = int(education == "7th-8th")
feature_values["education_9th"] = int(education == "9th")
feature_values["education_Assoc-acdm"] = int(education == "Associate Degree (Academic)")
feature_values["education_Assoc-voc"] = int(education == "Associate Degree (Vocational/Technical)")
feature_values["education_Bachelors"] = int(education == "Bachelors")
feature_values["education_Doctorate"] = int(education == "Doctorate")
feature_values["education_HS-grad"] = int(education == "HS-grad")
feature_values["education_Masters"] = int(education == "Masters")
feature_values["education_Preschool"] = int(education == "Preschool")
feature_values["education_Prof-school"] = int(education == "Prof-school")
feature_values["education_Some-college"] = int(education == "Some-college")


feature_values["marital.status_Married-AF-spouse"] = int(marital_status == "married(spouse is in armed force)")
feature_values["marital.status_Married-civ-spouse"] = int(marital_status == "married")
feature_values["marital.status_Married-spouse-absent"] = int(marital_status == "married(spouse live elsewhere)")
feature_values["marital.status_Never-married"] = int(marital_status == "not-married")
feature_values["marital.status_Separated"] = int(marital_status == "seperated")
feature_values["marital.status_Widowed"] = int(marital_status == "widowed")

feature_values["occupation_Armed-Forces"] = int(occupation == "Armed-Forces")
feature_values["occupation_Craft-repair"] = int(occupation == "Craft-repair")
feature_values["occupation_Exec-managerial"] = int(occupation == "Executive-managerial")
feature_values["occupation_Farming-fishing"] = int(occupation == "Farming-fishing")
feature_values["occupation_Handlers-cleaners"] = int(occupation == "Handlers-cleaners")
feature_values["occupation_Machine-op-inspct"] = int(occupation == "Machine-operator-inspector")
feature_values["occupation_Other-service"] = int(occupation == "Other-service")
feature_values["occupation_Priv-house-serv"] = int(occupation == "Private-house-service")
feature_values["occupation_Prof-specialty"] = int(occupation == "Prof-specialty")
feature_values["occupation_Protective-serv"] = int(occupation == "Protective-service")
feature_values["occupation_Sales"] = int(occupation == "Sales")
feature_values["occupation_Tech-support"] = int(occupation == "Tech-support")
feature_values["occupation_Transport-moving"] = int(occupation == "Transport-moving")

feature_values["relationship_Not-in-family"] = int(relationship == "Not in family")
feature_values["relationship_Other-relative"] = int(relationship == "Other relative")
feature_values["relationship_Own-child"] = int(relationship == "Own child")
feature_values["relationship_Unmarried"] = int(relationship == "Unmarried")
feature_values["relationship_Wife"] = int(relationship == "Wife")

feature_values["race_Other"] = int(race == "other")
feature_values["race_White"] = int(race == "white")

feature_values["region_Asia"] = int(region == "Asia")
feature_values["region_Central America & Caribbean"] = int(region == "Central America & Caribbean")
feature_values["region_Europe"] = int(region == "Europe")
feature_values["region_North America"] = int(region == "North America")
feature_values["region_Oceania"] = int(region == "Oceania")
feature_values["region_South America"] = int(region == "South America")


if st.button("Predict"):
    input_data = pd.DataFrame([feature_values])

    with st.spinner("Predicting..."):
        time.sleep(2)
        probability = model.predict_proba(input_data)[0, 1]

    if probability >= 0.8:
        st.success("🎉 Your income is predicted to be greater than $50K.")
        st.balloons()
    else:
        st.error("Your income is predicted to be $50K or less.")

    st.write(f"Probability of earning more than $50K:  **{probability:.2%}**")

st.divider()
st.header("Made by:")
st.write("Aaman manzar")
st.markdown("### 🌐 Connect with Me")
st.markdown("[GitHub](https://github.com/donkedo)")
st.markdown("[LinkedIn](https://www.linkedin.com/in/aaman-manzar)")



st.sidebar.header("👋 About the Developer")

st.sidebar.markdown("""
**Aaman Manzar**

🎓 B.Tech Student

💡 Interested in:
- Machine Learning
- Data Science
- Backend Development

🛠️ Technologies:
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Git & GitHub

This application predicts whether an individual's annual income is likely to exceed **$50,000** based on demographic and employment-related information.
""")
st.sidebar.markdown("[GitHub](https://github.com/donkedo)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/aaman-manzar)")