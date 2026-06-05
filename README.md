# 🌸 Iris Flower Classification using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

# 📌 Project Overview

The Iris Flower Classification project is a Machine Learning application that predicts the species of an Iris flower based on its physical measurements.

The model is trained using the famous Iris dataset from Scikit-Learn and classifies flowers into the following categories:

- Setosa
- Versicolor
- Virginica

This project demonstrates the complete Machine Learning lifecycle, including:

✅ Data Collection  
✅ Data Exploration  
✅ Data Visualization  
✅ Data Preprocessing  
✅ Model Training  
✅ Model Evaluation  
✅ Model Serialization (Pickle)  
✅ Streamlit Deployment

---

# 🌐 Live Demo

🚀 **Try the Application Here**

https://mhm8lmnoncnkfyxvodpmis.streamlit.app/

The web application allows users to enter flower measurements and instantly predict the Iris flower species using the trained Machine Learning model.

---

# 📂 Project Structure

```text
Iris-Flower-Classification/
│
├── Iris.ipynb              # Jupyter Notebook containing complete workflow
├── iris.pkl                # Trained Machine Learning model
├── README.md               # Project Documentation
├── requirements.txt        # Required dependencies
└── app.py                  # Streamlit Application
```

---

# 📊 Dataset Information

The Iris dataset contains 150 flower samples and 4 numerical features.

| Feature | Description |
|----------|------------|
| Sepal Length | Length of sepal in cm |
| Sepal Width | Width of sepal in cm |
| Petal Length | Length of petal in cm |
| Petal Width | Width of petal in cm |

### Target Classes

| Label | Species |
|--------|----------|
| 0 | Setosa |
| 1 | Versicolor |
| 2 | Virginica |

---

# 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Pickle
- Streamlit

---

# ⚙️ Machine Learning Workflow

## 1. Data Loading

Load the Iris dataset using Scikit-Learn.

## 2. Exploratory Data Analysis (EDA)

- Dataset Overview
- Feature Analysis
- Statistical Summary
- Class Distribution

## 3. Data Visualization

- Box Plots
- Scatter Plots
- Feature Relationships

## 4. Data Preprocessing

- Feature Selection
- Data Splitting
- Data Validation

## 5. Model Training

A Decision Tree Classifier is used for classification.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)
```

## 6. Model Evaluation

The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Precision Score
- Recall Score
- F1 Score
- ROC-AUC Score

## 7. Model Saving

```python
import pickle

with open("iris.pkl", "wb") as file:
    pickle.dump(model, file)
```

---

# 📈 Model Features

- Fast Predictions
- User-Friendly Interface
- Real-Time Classification
- Lightweight Deployment
- High Accuracy Classification

---

# 🚀 How to Run Locally

## Clone Repository

```bash
git clone https://github.com/Prasanna99-rgb/Iris-Flower-Classification.git
```

## Navigate to Project Directory

```bash
cd Iris-Flower-Classification
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit Application

```bash
streamlit run app.py
```

---

# 📦 Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
streamlit
jupyter
```

---

# 🎯 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Random Forest Classifier
- XGBoost Classifier
- Model Monitoring
- Cloud Deployment

---

# 👨‍💻 Author

## Prasanna Deshmane

🔗 GitHub: https://github.com/Prasanna99-rgb

🔗 LinkedIn: https://www.linkedin.com/in/prasanna-deshmane-80a419205

🌐 Live Demo: https://mhm8lmnoncnkfyxvodpmis.streamlit.app/

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork this repository and submit pull requests.

---

# ⭐ Support

If you found this project useful, please give it a ⭐ on GitHub.

---

## Made with ❤️ by Prasanna Deshmane

---
Made with ❤️ using Python and Scikit-Learn
