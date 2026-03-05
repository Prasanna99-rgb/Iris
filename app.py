import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("iris.pkl", "rb"))

st.title("🌸 Iris Flower Prediction App")

st.write("Enter the flower measurements to predict the species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0)

# Prediction button
if st.button("Predict Species"):

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(features)

    species = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Iris Species: **{species[prediction[0]]}**")
