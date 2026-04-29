import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Placement Prediction", layout="centered")

st.title("🎓 Placement Prediction System (Computer Dept)")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload Placement Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset loaded!")
else:
    df = pd.read_csv("data.csv")

# ---------------- VALIDATION ----------------
required_cols = ["Year", "Total", "Placed", "AvgPackage", "Companies"]

if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain: Year, Total, Placed, AvgPackage, Companies")
    st.stop()

# Filter Computer Dept if exists
if "Dept" in df.columns:
    df = df[df["Dept"] == "Computer"]

# ---------------- SHOW DATA ----------------
st.subheader("📊 Dataset Preview")
st.dataframe(df)

# ---------------- MODEL TRAINING ----------------
X = df[["Year", "Total"]]
y = df["Placed"]

# Linear
lr = LinearRegression()
lr.fit(X, y)

# Polynomial
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

pr = LinearRegression()
pr.fit(X_poly, y)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Salary model
salary_model = RandomForestRegressor()
salary_model.fit(X, df["AvgPackage"])

# Classification
df["PlacedFlag"] = (df["Placed"] / df["Total"]) >= 0.75

if df["PlacedFlag"].nunique() < 2:
    df["PlacedFlag"] = df["Placed"] >= df["Placed"].median()

clf = LogisticRegression()
clf.fit(X, df["PlacedFlag"])

# ---------------- INPUT ----------------
year = st.number_input("Enter Year", min_value=2024, max_value=2035)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    input_data = pd.DataFrame([[year, 60]], columns=["Year", "Total"])

    lr_pred = lr.predict(input_data)[0]
    pr_pred = pr.predict(poly.transform(input_data))[0]
    rf_pred = rf.predict(input_data)[0]
    salary_pred = salary_model.predict(input_data)[0]
    status = clf.predict(input_data)[0]

    # Scores
    scores = {
        "Linear": r2_score(y, lr.predict(X)),
        "Polynomial": r2_score(y, pr.predict(X_poly)),
        "RandomForest": r2_score(y, rf.predict(X))
    }

    best_model = max(scores, key=scores.get)

    if best_model == "Linear":
        best_value = lr_pred
    elif best_model == "Polynomial":
        best_value = pr_pred
    else:
        best_value = rf_pred

    # ---------------- OUTPUT ----------------
    st.subheader("📊 Model Predictions")
    st.write("Linear:", int(lr_pred))
    st.write("Polynomial:", int(pr_pred))
    st.write("Random Forest:", int(rf_pred))

    st.subheader("🤖 Best Model Prediction")
    st.success(f"Selected Model: {best_model}")
    st.write("Predicted Students Placed:", int(best_value))

    st.write("💰 Avg Salary:", round(salary_pred, 2))

    if status:
        st.success("High Placement Expected")
    else:
        st.warning("Moderate Placement Expected")

# ---------------- MODEL COMPARISON ----------------
st.subheader("📈 Model Comparison")

scores = {
    "Linear": r2_score(y, lr.predict(X)),
    "Polynomial": r2_score(y, pr.predict(X_poly)),
    "RandomForest": r2_score(y, rf.predict(X))
}

st.write(scores)

plt.figure()
plt.bar(scores.keys(), scores.values())
st.pyplot(plt)

# ---------------- GRAPHS ----------------
st.subheader("📉 Placement Trend")

plt.figure()
plt.plot(df["Year"], df["Placed"], marker='o')
plt.xlabel("Year")
plt.ylabel("Students Placed")
st.pyplot(plt)

st.subheader("🏢 Companies Visited Trend")

plt.figure()
plt.plot(df["Year"], df["Companies"], marker='o')
plt.xlabel("Year")
plt.ylabel("Companies")
st.pyplot(plt)
