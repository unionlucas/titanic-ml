import streamlit as st
import joblib
import pandas as pd
from PIL import Image


model = joblib.load("./data/model.pkl")
feature_names = joblib.load("./data/feature_names.pkl")
def predict(p_class : int, age : int, sex : int, spouse_size : int, parch: int, ticket_price : float, departure: str):
    sex_encoded = 0
    if sex == "female":
        sex_encoded = 1
    
    q = 0
    s = 0
    if departure == "Southhampton (England)":
        s = 1
    if departure == "Queenstown (Ireland)":
        q = 1


    input_df = pd.DataFrame([{
        "Pclass": p_class,
        "Sex": sex_encoded,
        "Age": age,
        "SibSp": spouse_size,
        "Parch": parch,
        "Fare": ticket_price,
        "Embarked_Q": q,
        "Embarked_S": s
    }])
    prediction = model.predict(input_df)
    return prediction[0]

# ---- Streamlit ----
st.set_page_config(page_title="Titanic Prediction", page_icon="🚢")
st.title("Titanic Prediction")

with st.expander("Input your data", expanded=True):
    with st.form("in"):
        st.write("Your Data")
        age_val = st.slider("Age", 1, 100, value=29)
        sex = st.radio("Sex", ("female", "male"), horizontal=True, index = 1)
        p_class = st.radio("Passenger Class", (1, 2, 3), horizontal=True, index = 1)
        spouse_size = st.slider("Siblings and Spouses", 0, 10, value=1)
        parch = st.slider("Parents and Children", 0, 10, value=0)
        ticket_price = st.slider("Ticket Price (£)", 0, 600, value=32)
        departure = st.selectbox("Departure", options=["Cherbourg (France)", "Southhampton (England)", "Queenstown (Ireland)"], index=0)

        submitted = st.form_submit_button("Submit")
        
if submitted:
    result = predict(p_class, age_val, sex, spouse_size, parch, ticket_price, departure)
    if result == 0:
        st.error("Not survived!")
    else:
        st.success("Survived!")
with st.expander("Metrics / Charts", expanded=False):
    conf_matrix_img = Image.open("./data/confusion_matrix.png")
    roc_curve_img = Image.open("./data/roc_curve.png")
    correlation_img = Image.open("./data/correlations.png")

    # Bilder in Streamlit anzeigen
    st.header("Confusion Matrix")
    st.image(conf_matrix_img)

    st.header("ROC-Logisitc")
    st.image(roc_curve_img)

    st.header("Correlations")
    st.image(correlation_img)


