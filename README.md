# 🌸 Iris Flower Classification using Decision Tree

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Overview

This project implements a Machine Learning model to classify Iris flowers into three different species:

- Setosa
- Versicolor
- Virginica

The model is trained using the famous Iris Dataset from Scikit-Learn and demonstrates the complete Machine Learning workflow from data preprocessing to model deployment.

---

---
## 🌐 Live Demo

🔗 **Streamlit App:** https://mhm8lmnoncnkfyxvodpmis.streamlit.app/

Explore the live application to predict Iris flower species based on sepal and petal measurements using the trained Decision Tree model.

---

## 📂 Project Structure

```text
Iris-Flower-Classification/
│
├── Iris.ipynb
├── iris.pkl
├── README.md
└── requirements.txt
```

---

## 📊 Dataset Information

The Iris dataset contains 150 observations and 4 features.

| Feature | Description |
|----------|------------|
| Sepal Length | Length of sepal (cm) |
| Sepal Width | Width of sepal (cm) |
| Petal Length | Length of petal (cm) |
| Petal Width | Width of petal (cm) |

### Target Classes

| Class | Species |
|---------|----------|
| 0 | Setosa |
| 1 | Versicolor |
| 2 | Virginica |

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Pickle

---

## ⚙️ Project Workflow

### 1. Data Loading
Load the Iris dataset using Scikit-Learn.

### 2. Exploratory Data Analysis
- Dataset Overview
- Statistical Summary
- Class Distribution

### 3. Data Visualization
- Boxplots
- Scatterplots
- Feature Relationships

### 4. Data Preprocessing
- Feature Selection
- Train-Test Split

### 5. Model Training
A Decision Tree Classifier is used for classification.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)
```

### 6. Model Evaluation

The model performance is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- ROC-AUC Score

### 7. Model Saving

```python
import pickle

with open("iris.pkl", "wb") as file:
    pickle.dump(model, file)
```

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/your-github-username/Iris-Flower-Classification.git
```

### Move to Project Directory

```bash
cd Iris-Flower-Classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Jupyter Notebook

```bash
jupyter notebook
```

Open `Iris.ipynb` and run all cells.

---

## 📦 Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🎯 Future Enhancements

- Hyperparameter Tuning
- Cross Validation
- Random Forest Classifier
- XGBoost Classifier
- Streamlit Deployment

---

## 👨‍💻 Author

### Prasanna Deshmane

🔗 LinkedIn: https://www.linkedin.com/in/prasanna-deshmane-80a419205

🔗 GitHub: https://github.com/your-github-username

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---
Made with ❤️ using Python and Scikit-Learn
