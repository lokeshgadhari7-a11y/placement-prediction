import streamlit as st
from model import predict_best

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Placement Prediction", layout="centered")

# -------------------------------
# Title
# -------------------------------
st.title("🎯 Placement Prediction System")
st.markdown("Predict placement chances based on historical data")

# -------------------------------
# Input Section
# -------------------------------
st.subheader("📊 Select Year")

year = st.slider("Select Academic Year", 2015, 2025, 2022)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Placement"):

    value, best_model = predict_best(year)

    st.subheader("📌 Result")

    if value > 0.75:
        st.success("✅ High placement rate expected")
    else:
        st.error("❌ Lower placement rate expected")

    st.info(f"🤖 Best Model Used: {best_model}")
    st.write(f"📈 Predicted Placement Value: {round(value, 2)}")

# -------------------------------
# About Section
# -------------------------------
st.markdown("---")
st.subheader("📖 About This Project")

st.write("""
- Uses Machine Learning models to predict placement trends
- Models used: Linear, Polynomial, Random Forest
- Built using Python, Scikit-learn, and Streamlit
""")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by Lokesh Gadhari 🚀")
