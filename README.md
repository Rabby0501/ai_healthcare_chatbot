# 🩺 AI Healthcare ChatBot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered healthcare chatbot that predicts possible diseases based on user symptoms using a Decision Tree classifier. The project includes both a terminal version and a modern **Streamlit web interface** with visual severity indicators, confidence scores, PDF report generation, and consultation history.

> ⚠️ **Medical Disclaimer** – This tool is for educational purposes only. Always consult a qualified healthcare professional for medical advice.

---

## ✨ Features

- **Symptom-based disease prediction** – Select from 132 symptoms, covers 42 diseases.
- **Confidence score** – Shows how certain the model is about each prediction.
- **Severity visualization** – Progress bar + color-coded severity level (Mild/Moderate/Severe).
- **Disease description & precautions** – Detailed information for each predicted condition.
- **PDF report download** – Generate a complete consultation report for your doctor.
- **Consultation history** – Sidebar stores past sessions for easy reference.
- **Medical disclaimer** – Prominent warning in the sidebar.
- **Two interfaces** – Terminal-based (`chat_bot.py`) and Streamlit web app (`streamlit_app.py`).

---

## 📊 Dataset

The project uses two complementary datasets from Kaggle:

| Dataset | File(s) | Purpose |
|---------|---------|---------|
| **Disease Prediction** | `Data/Training.csv`, `Data/Testing.csv` | 132 symptom columns (binary) + 1 prognosis column. 42 diseases. |
| **Symptom Metadata** | `MasterData/Symptom_severity.csv`, `symptom_Description.csv`, `symptom_precaution.csv` | Severity weights, descriptions, and precautions for each disease. |

> The model is trained on `Training.csv` and evaluated on `Testing.csv`. Accuracy on test set: **~97.5%**.

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Machine Learning:** scikit-learn (DecisionTreeClassifier, SVM)
- **Data Handling:** pandas, numpy, csv
- **Web Interface:** Streamlit
- **PDF Generation:** FPDF
- **Other:** pyttsx3 (text-to-speech in terminal version)

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-healthcare-chatbot.git
cd ai-healthcare-chatbot
