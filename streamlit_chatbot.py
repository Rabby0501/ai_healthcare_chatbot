import streamlit as st
import pandas as pd
import numpy as np
import csv
import io
from fpdf import FPDF
from datetime import datetime
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ------------------ Page config ------------------
st.set_page_config(page_title="AI Healthcare ChatBot", page_icon="🩺", layout="wide")

# ------------------ Load and train model (cached) ------------------
@st.cache_resource
def load_model():
    training = pd.read_csv('Data/Training.csv')
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']
    
    le = preprocessing.LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    x_train, _, y_train, _ = train_test_split(x, y_encoded, test_size=0.33, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    
    # Also train for probability (confidence)
    clf_proba = DecisionTreeClassifier()
    clf_proba.fit(x_train, y_train)
    
    return clf, clf_proba, le, cols, training

@st.cache_data
def load_severity():
    severity = {}
    try:
        with open('MasterData/Symptom_severity.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    severity[row[0].strip()] = int(row[1].strip())
    except FileNotFoundError:
        st.error("Severity file not found.")
    return severity

@st.cache_data
def load_description():
    desc = {}
    try:
        with open('MasterData/symptom_Description.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    desc[row[0].strip()] = row[1].strip()
    except FileNotFoundError:
        st.error("Description file not found.")
    return desc

@st.cache_data
def load_precautions():
    prec = {}
    try:
        with open('MasterData/symptom_precaution.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5:
                    prec[row[0].strip()] = [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()]
    except FileNotFoundError:
        st.error("Precaution file not found.")
    return prec

# ------------------ Helper functions ------------------
def calc_severity_score(symptoms, severity_dict):
    """Calculate total severity score."""
    return sum(severity_dict.get(symptom, 0) for symptom in symptoms)

def get_confidence(model, input_vector, le):
    """Get confidence score (max probability) from decision tree."""
    proba = model.predict_proba([input_vector])[0]
    confidence = max(proba) * 100
    return confidence

def predict_with_confidence(symptoms_exp, model, model_proba, le, training_df):
    """Predict disease and return confidence score."""
    X = training_df.iloc[:, :-1]
    symptoms_list = list(X.columns)
    input_vector = np.zeros(len(symptoms_list))
    for sym in symptoms_exp:
        if sym in symptoms_list:
            input_vector[symptoms_list.index(sym)] = 1
    
    pred_encoded = model.predict([input_vector])[0]
    disease = le.inverse_transform([pred_encoded])[0]
    confidence = get_confidence(model_proba, input_vector, le)
    return disease, confidence

def generate_pdf_report(name, symptoms, days, disease, confidence, description, precautions, severity_score):
    """Generate PDF report as bytes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Healthcare ChatBot - Consultation Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Symptoms:", ln=True)
    pdf.set_font("Arial", "", 12)
    for sym in symptoms:
        pdf.cell(0, 6, f"- {sym}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Duration: {days} days", ln=True)
    pdf.cell(0, 10, f"Severity Score: {severity_score}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Predicted Disease: {disease}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.1f}%", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Description:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, description)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Precautions:", ln=True)
    pdf.set_font("Arial", "", 12)
    for i, p in enumerate(precautions, 1):
        pdf.cell(0, 6, f"{i}. {p}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, "Disclaimer: This is an AI-generated report. Always consult a real doctor.", ln=True)
    
    return pdf.output(dest='S').encode('latin1')

# ------------------ Sidebar (Disclaimer & History) ------------------
with st.sidebar:
    st.header("⚠️ Medical Disclaimer")
    st.warning(
        "This chatbot is for **informational and educational purposes only**. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
    )
    st.markdown("---")
    
    st.header("📋 Consultation History")
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if st.session_state.history:
        for idx, record in enumerate(reversed(st.session_state.history[-5:])):  # show last 5
            with st.expander(f"{record['date']} - {record['disease']}"):
                st.write(f"**Symptoms:** {', '.join(record['symptoms'])}")
                st.write(f"**Confidence:** {record['confidence']:.1f}%")
                st.write(f"**Severity:** {record['severity_score']}")
        if len(st.session_state.history) > 5:
            st.caption(f"Showing last 5 of {len(st.session_state.history)} consultations")
    else:
        st.info("No consultations yet. Complete a diagnosis to see history here.")
    
    st.markdown("---")
    st.caption("Developed by Rabby Md Golam | AI Healthcare ChatBot")

# ------------------ Main UI ------------------
st.title("🩺 AI Healthcare ChatBot")
st.markdown("Describe your symptoms and I'll suggest possible conditions and precautions.")

# Load all data and model
clf, clf_proba, le, feature_cols, training_df = load_model()
severity_dict = load_severity()
description_dict = load_description()
precaution_dict = load_precautions()

# Session state initialization
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.name = ""
    st.session_state.symptoms = []
    st.session_state.days = 0
    st.session_state.disease = None
    st.session_state.confidence = 0
    st.session_state.prediction_done = False
    st.session_state.report_generated = False

# Step 1: Ask for name
if st.session_state.step == 1:
    name = st.text_input("What is your name?")
    if st.button("Start Consultation"):
        if name:
            st.session_state.name = name
            st.session_state.step = 2
            st.rerun()
        else:
            st.warning("Please enter your name.")

# Step 2: Symptom selection
elif st.session_state.step == 2:
    st.write(f"Hello **{st.session_state.name}**, please add the symptoms you are experiencing.")
    
    symptom_options = sorted(feature_cols.tolist())
    selected_symptom = st.selectbox("Select a symptom:", symptom_options)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add symptom"):
            if selected_symptom and selected_symptom not in st.session_state.symptoms:
                st.session_state.symptoms.append(selected_symptom)
                st.success(f"Added: {selected_symptom}")
            elif selected_symptom in st.session_state.symptoms:
                st.warning("Symptom already added.")
            else:
                st.warning("Select a symptom first.")
    
    with col2:
        if st.button("🗑 Remove last symptom"):
            if st.session_state.symptoms:
                removed = st.session_state.symptoms.pop()
                st.info(f"Removed: {removed}")
            else:
                st.warning("No symptoms to remove.")
    
    st.write("**Your current symptoms:**")
    if st.session_state.symptoms:
        for sym in st.session_state.symptoms:
            st.write(f"- {sym}")
    else:
        st.write("_None added yet._")
    
    if st.button("✅ I have finished adding symptoms"):
        if len(st.session_state.symptoms) > 0:
            st.session_state.step = 3
            st.rerun()
        else:
            st.warning("Please add at least one symptom before proceeding.")

# Step 3: Number of days
elif st.session_state.step == 3:
    st.write(f"**{st.session_state.name}**, for how many days have you been experiencing these symptoms?")
    days = st.number_input("Number of days:", min_value=1, max_value=365, step=1, value=1)
    if st.button("Analyze Symptoms"):
        st.session_state.days = days
        st.session_state.step = 4
        st.rerun()

# Step 4: Prediction and results
elif st.session_state.step == 4:
    st.subheader("🩺 Analysis Results")
    
    with st.spinner("Analyzing your symptoms..."):
        # Predict disease with confidence
        predicted, confidence = predict_with_confidence(
            st.session_state.symptoms, clf, clf_proba, le, training_df
        )
        st.session_state.disease = predicted
        st.session_state.confidence = confidence
        
        # Severity score
        severity_score = calc_severity_score(st.session_state.symptoms, severity_dict)
        
        # Visual severity meter
        st.subheader("📊 Symptom Severity")
        severity_percent = min(severity_score / 100, 1.0)  # cap at 100
        st.progress(severity_percent)
        if severity_score < 30:
            st.success(f"🟢 Severity Score: {severity_score} (Mild)")
        elif severity_score < 70:
            st.warning(f"🟡 Severity Score: {severity_score} (Moderate)")
        else:
            st.error(f"🔴 Severity Score: {severity_score} (Severe)")
        
        # Disease and confidence
        st.success(f"**You may have:** {predicted}")
        st.metric("Confidence Score", f"{confidence:.1f}%")
        
        # Description
        desc_text = description_dict.get(predicted, "No description available.")
        st.write(f"**Description:** {desc_text}")
        
        # Precautions
        st.subheader("📋 Recommended Precautions")
        precautions = precaution_dict.get(predicted, ["No specific precautions listed."])
        for i, p in enumerate(precautions, 1):
            st.write(f"{i}. {p}")
        
        # Severity advice based on both score and duration
        total_severity_impact = (severity_score * st.session_state.days) / (len(st.session_state.symptoms) + 1)
        if total_severity_impact > 13:
            st.error("⚠️ **High severity detected.** Please consult a doctor immediately.")
        else:
            st.info("ℹ️ Based on your symptoms, it's advisable to monitor and consult a doctor if symptoms persist.")
        
        # Save to history
        history_record = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "symptoms": st.session_state.symptoms.copy(),
            "days": st.session_state.days,
            "disease": predicted,
            "confidence": confidence,
            "severity_score": severity_score
        }
        st.session_state.history.append(history_record)
        
        # PDF Report button
        st.subheader("📄 Generate Report")
        if st.button("🖨️ Print / Download PDF Report"):
            pdf_bytes = generate_pdf_report(
                st.session_state.name,
                st.session_state.symptoms,
                st.session_state.days,
                predicted,
                confidence,
                desc_text,
                precautions,
                severity_score
            )
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"health_report_{st.session_state.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        
        # Option to start over
        if st.button("🔄 Start Over"):
            for key in list(st.session_state.keys()):
                if key not in ['history']:  # keep history across sessions
                    del st.session_state[key]
            st.session_state.step = 1
            st.session_state.symptoms = []
            st.session_state.prediction_done = False
            st.rerun()

else:
    st.write("Something went wrong. Please refresh the page.")