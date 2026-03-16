# 🎓 Predicting University Admission Chances

## Using Machine Learning — Linear Regression | Random Forest | XGBoost

**Designed & Developed by Engineer. Roaa Abu Arra**

[![Streamlit App](https://img.shields.io/badge/Live%20App-Click%20Here-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://university-admission.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

---

## 📌 Project Overview

This project is a complete end-to-end **Data Science & Machine Learning** application.

It predicts a student's probability of admission to a graduate university program based on their academic profile.

The system is deployed as a **bilingual (English/Arabic) interactive web application** built with Streamlit.

---

## 🌐 Live Demo

🔗 [university-admission.streamlit.app](https://university-admission.streamlit.app)

---

## 📊 Dataset

- **Records:** 500 students
- **Features:** GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research Experience
- **Target:** Chance of Admission (0.0 – 1.0)

---

## 🤖 Machine Learning Models

| Model | Description | R² Score |
|-------|-------------|----------|
| Linear Regression | Simple and fast baseline model | ~0.82 |
| Random Forest | Ensemble of 100 decision trees | ~0.87 |
| **XGBoost** ⭐ | **Best model — gradient boosting** | **~0.89** |

---

## 🖥️ App Features

- 🌐 **Bilingual** — English and Arabic with one-click toggle
- 🎛️ **Interactive Sidebar** — Sliders and dropdowns for all inputs
- 🏫 **University & Major Selection** — 8 top universities and 8 fields of study
- 📝 **Feature Descriptions** — Clear explanations for GRE, TOEFL, SOP, LOR, CGPA
- 🤖 **Model Selection** — Choose between 3 ML models with plain-language descriptions

### 📑 Three Tabs

| Tab | Content |
|-----|---------|
| 📊 Data & Visualization | Dataset preview, charts, correlation heatmap |
| 📈 Model Performance | Comparison table, R² and MSE charts, feature importance |
| 🔮 Prediction | Instant result with progress bar and all-models comparison |

---

## 🚀 Run Locally

```bash
git clone https://github.com/RoaaRaed/admission-app.git
cd admission-app
pip install -r requirements.txt
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## 📁 Repository Structure

```
admission-app/
├── app.py                                   # Main Streamlit application
├── requirements.txt                         # Python dependencies
└── University_Admission_Documentation.pdf  # Full project documentation
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
```

---

## 📈 Key Results

- **Best Model:** XGBoost with R² ≈ 0.89
- **Most Important Feature:** CGPA (correlation = 0.87)
- **Second Most Important:** GRE Score (correlation = 0.80)

---

## 👩‍💻 Author

**Engineer. Roaa Abu Arra**

- GitHub: [@RoaaRaed](https://github.com/RoaaRaed)
- Live App: [university-admission.streamlit.app](https://university-admission.streamlit.app)

---

*Data Science Project — 2026*
