
# ❤️ Heart Disease Prediction Using Machine Learning

This project predicts whether a patient has heart disease (`0 = No`, `1 = Yes`) using multiple machine learning classification models.  
It includes model training, evaluation, comparison, and a Streamlit web application for real-time predictions.

---

## 📌 Project Overview

Heart disease is a major cause of death worldwide. Early prediction can help in timely treatment and reduce risks.  
This project uses machine learning algorithms to predict heart disease using medical attributes such as:

- Age  
- Sex  
- Chest Pain Type  
- Blood Pressure  
- Cholesterol  
- ECG Results  
- Maximum Heart Rate  
- Exercise-induced Angina  
- Oldpeak  
- Number of Vessels  
- Thalassemia  

The target variable is:
- **1 = Heart Disease present**
- **0 = No Heart Disease**

---

## 🧠 Models Used

The following ML classifiers are trained and compared:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Neural Network (MLP Classifier)

Models are evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

---

## 📊 Features

- Model training & evaluation (`major_project.py`)  
- ROC Curve visualization  
- Feature Importance analysis  
- Streamlit App for predictions (`app.py`)  
- Saved model files (`rf_model.pkl`, `scaler.pkl`)  
- PDF report included  

---

## 🚀 How to Run the Training Script

To train and evaluate models, run:

```bash
python major_project.py
```

This script trains all machine learning models, evaluates them, plots ROC curves, and saves:

- `rf_model.pkl` (trained Random Forest model)
- `scaler.pkl` (StandardScaler used for preprocessing)

---

## 🩺 How to Use the Prediction App

To launch the Streamlit app and predict heart disease risk, run:

```bash
streamlit run app.py
```

This will open a browser window at: [http://localhost:8501/](http://localhost:8501/)

The app allows you to input patient attributes:

- Age  
- Sex  
- Chest pain type  
- Blood pressure  
- Cholesterol  
- ECG results  
- Max heart rate  
- Exercise-induced angina  
- Oldpeak  
- Slope  
- Number of vessels  
- Thalassemia  

After clicking **Predict**, you will see:

- ❤️ **Low chance of heart disease**
- 💔 **High chance of heart disease**
- Probability score (0–1)

---

## 📁 Project Structure

```
Heart-Disease-Predictor/
│
├── app.py             # Streamlit prediction app
├── major_project.py   # Training & evaluation script
├── heart.csv          # Dataset
├── rf_model.pkl       # Saved Random Forest model
├── scaler.pkl         # Saved StandardScaler
├── requirements.txt   # Dependencies
├── report.pdf         # Final project report
└── README.md          # Documentation
```

---

## 📄 Requirements

Make sure to install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 📬 Contact

For any questions or feedback, feel free to open an issue or reach out via [GitHub](https://github.com/Mayuur-web).

---
````

