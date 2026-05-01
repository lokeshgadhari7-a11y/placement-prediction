import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Placement Prediction", layout="centered")

# -------------------------------
# Title
# -------------------------------
st.title("🎯 Placement Prediction System")
st.markdown("Predict your chances of getting placed based on academic performance.")

# -------------------------------
# Load Model
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))

# -------------------------------
# Input Section
# -------------------------------
st.subheader("📊 Enter Your Details")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
iq = st.slider("IQ Score", 0, 200, 100)
profile_score = st.slider("Profile Score (Projects/Skills)", 0, 100, 50)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Placement Chance"):

    input_data = np.array([[cgpa, iq, profile_score]])
    prediction = model.predict(input_data)

    st.subheader("📌 Result")

    if prediction[0] == 1:
        st.success("✅ You have a HIGH chance of getting placed!")
    else:
        st.error("❌ You have a LOW chance of getting placed.")

# -------------------------------
# About Section
# -------------------------------
st.markdown("---")
st.subheader("📖 About This Project")

st.write("""
- This project uses Machine Learning to predict placement chances.
- Algorithms used: Linear Regression, Logistic Regression.
- Built using Python, Scikit-learn, and Streamlit.
- Developed as part of a data science learning project.
""")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by Lokesh Gadhari 🚀")
