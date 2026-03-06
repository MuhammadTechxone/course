import streamlit as st
import pandas as pd
import joblib

model = joblib.load("psychotherapy_svm_model.pkl")
st.title = ("Psychotherapy Assistant")
User_prompt = st.text_input("Enter your text to be analyzed: ")
if st.button("submit"):
    data = pd.DataFrame({"text":[User_prompt]}
        )
    prediction = model.predict(data)
    predicted_new = prediction[0]
    st.success(f"Recommended course: {predicted_new}")
    