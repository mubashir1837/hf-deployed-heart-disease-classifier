---
title: Heart Disease Classifier
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: "1.41.1"
app_file: app.py
pinned: true
license: mit
tags:
  - heart-disease
  - classification
  - random-forest
  - healthcare
  - machine-learning
datasets:
  - mubashir1837/heart-disease
short_description: ML heart disease risk prediction dashboard
---

# Heart Disease Classifier 🫀

A complete, production-quality Machine Learning pipeline that trains, evaluates, and serves a heart disease prediction model using the [Mubashir1837 Heart Disease dataset](https://huggingface.co/datasets/mubashir1837/heart-disease) from Hugging Face.

## 🚀 Live Demo

This Space provides an **interactive Streamlit dashboard** with 4 tabs:

| Tab | Description |
|-----|-------------|
| 🔍 **Prediction** | Enter patient data via sliders → get real-time disease probability with gauge chart |
| 📊 **EDA Dashboard** | Target distribution, feature histograms, correlation heatmap, boxplots, pairplots |
| 📈 **Model Performance** | Accuracy, ROC-AUC, confusion matrix, ROC curve, cross-validation comparison |
| 🎯 **Feature Importance** | Interactive Plotly chart of top feature importances with reference guide |

## Model Details

| Metric | Value |
|--------|-------|
| **Algorithm** | Random Forest (GridSearchCV-tuned) |
| **Accuracy** | 78.69% |
| **ROC-AUC** | 0.8983 |
| **Macro F1** | 0.7783 |
| **Dataset** | 303 patients, 14 features |
| **CV Strategy** | 5-Fold Stratified Cross-Validation |

## Project Structure

```
heart-disease-classifier/
├── app.py                   ← Streamlit GUI (HF Spaces entry point)
├── main.py                  ← Full pipeline entry-point
├── requirements.txt
├── src/
│   ├── data_loader.py       ← Load dataset from HF Hub
│   ├── eda.py               ← Exploratory Data Analysis & plots
│   ├── preprocessor.py      ← Feature engineering + sklearn pipeline
│   ├── train.py             ← Multi-model training + GridSearch tuning
│   └── predict.py           ← Inference CLI for new patients
├── models/                  ← Saved model + metrics
└── plots/                   ← All generated visualisations
```

## Pipeline Features

| Stage | Details |
|---|---|
| **Data Loading** | Reads directly from `hf://` Hugging Face URL via pandas |
| **EDA** | 5 plots: target distribution, feature histograms, correlation heatmap, box-plots, pair-plot |
| **Feature Engineering** | Age bands, cholesterol flags, ST-depression severity, tachycardia flag |
| **Preprocessing** | StandardScaler + OneHotEncoder via `ColumnTransformer` |
| **Models Compared** | Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, XGBoost |
| **Selection** | 5-fold Stratified CV by ROC-AUC → auto-selects best |
| **Tuning** | GridSearchCV for the winning model |
| **Evaluation** | Accuracy, ROC-AUC, classification report, confusion matrix, ROC curve |
| **Inference** | CLI to predict on new patients (single or batch CSV) |

## Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the full pipeline
python main.py

# 3. Launch the Streamlit dashboard
streamlit run app.py

# 4. Predict on a new patient (CLI)
python src/predict.py --patient "63,1,3,145,233,1,0,150,0,2.3,0,0,1"
```

## Dataset Features

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dL) |
| `fbs` | Fasting blood sugar > 120 mg/dL |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels (0–4) |
| `thal` | Thalassemia type |
| `target` | **1** = Heart disease, **0** = No disease |

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical decisions.
