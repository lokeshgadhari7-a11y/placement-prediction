import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("data.csv")
df = df[df["Dept"] == "Computer"]

# Features
X = df[["Year", "Total"]]
y = df["Placed"]

# ---------------- MODELS ----------------
lr = LinearRegression()
lr.fit(X, y)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

pr = LinearRegression()
pr.fit(X_poly, y)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Salary model
salary_model = RandomForestRegressor()
salary_model.fit(X, df["AvgPackage"])

# ---------------- CLASSIFICATION ----------------
df["PlacedFlag"] = (df["Placed"] / df["Total"]) >= 0.75

if df["PlacedFlag"].nunique() < 2:
    df["PlacedFlag"] = df["Placed"] >= df["Placed"].median()

clf = LogisticRegression()
clf.fit(X, df["PlacedFlag"])

# ---------------- SCORES ----------------
def get_scores():
    return {
        "Linear": r2_score(y, lr.predict(X)),
        "Polynomial": r2_score(y, pr.predict(X_poly)),
        "RandomForest": r2_score(y, rf.predict(X))
    }

# ---------------- BEST MODEL ----------------
def best_model_name():
    scores = get_scores()
    best = max(scores, key=scores.get)
    return best, scores

# ---------------- PREDICT ALL ----------------
def predict(year):
    data = pd.DataFrame([[year, 60]], columns=["Year", "Total"])

    return {
        "Linear": lr.predict(data)[0],
        "Polynomial": pr.predict(poly.transform(data))[0],
        "RandomForest": rf.predict(data)[0],
        "Salary": salary_model.predict(data)[0],
        "Status": clf.predict(data)[0]
    }

# ---------------- PREDICT BEST ----------------
def predict_best(year):
    data = pd.DataFrame([[year, 60]], columns=["Year", "Total"])
    best, scores = best_model_name()

    if best == "Linear":
        value = lr.predict(data)[0]
    elif best == "Polynomial":
        value = pr.predict(poly.transform(data))[0]
    else:
        value = rf.predict(data)[0]

    return value, best, scores