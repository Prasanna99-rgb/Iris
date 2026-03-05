import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("iris.pkl", "rb"))

st.title("Iris Flower Prediction App")

st.write("Enter flower measurements")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(features)

    st.success(f"Predicted Iris Species: {prediction[0]}")
