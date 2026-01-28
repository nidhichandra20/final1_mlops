import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")

st.title("Iris Flower Classifier")

sl = st.number_input("Sepal length (cm)")
sw = st.number_input("Sepal width (cm)")
pl = st.number_input("Petal length (cm)")
pw = st.number_input("Petal width (cm)")

if st.button("Predict"):
    data = pd.DataFrame([[sl, sw, pl, pw]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ])
    prediction = model.predict(data)
    st.write("Prediction:", prediction)
