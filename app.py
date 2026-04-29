import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import predict, get_scores, predict_best

st.title("🎓 Placement Prediction System (Computer Dept)")

# Input
year = st.number_input("Enter Year", min_value=2024, max_value=2035)

if st.button("Predict"):

    result = predict(year)

    st.subheader("📊 Individual Model Predictions")
    st.write("Linear Model Prediction:", int(result["Linear"]))
    st.write("Polynomial Model Prediction:", int(result["Polynomial"]))
    st.write("Random Forest Model Prediction:", int(result["RandomForest"]))

    # Best model
    best_value, best_model, scores = predict_best(year)

    st.subheader("🤖 Best Model Prediction")
    st.success(f"Selected Model: {best_model}")
    st.write("Predicted Students Placed:", int(best_value))

    st.write("💰 Avg Salary:", round(result["Salary"], 2))

    if result["Status"]:
        st.success("High Placement Expected")
    else:
        st.warning("Moderate Placement Expected")

# ---------------- MODEL COMPARISON ----------------
st.subheader("📈 Model Comparison")
scores = get_scores()

st.write(scores)

plt.figure()
plt.bar(scores.keys(), scores.values())
st.pyplot(plt)

best_model = max(scores, key=scores.get)
st.info(f"Best Model based on R² Score: {best_model}")

# ---------------- DATA VISUALIZATION ----------------
df = pd.read_csv("data.csv")
df = df[df["Dept"] == "Computer"]

# Placement graph
st.subheader("📉 Placement Trend")
plt.figure()
plt.plot(df["Year"], df["Placed"], marker='o')
plt.xlabel("Year")
plt.ylabel("Students Placed")
st.pyplot(plt)

# Companies graph
st.subheader("🏢 Companies Visited Trend")
plt.figure()
plt.plot(df["Year"], df["Companies"], marker='o')
plt.xlabel("Year")
plt.ylabel("Companies")
st.pyplot(plt)
