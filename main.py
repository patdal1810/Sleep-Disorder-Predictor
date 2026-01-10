import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

if ARTIFACTS_DIR.exists() == False:
    os.mkdir('artifacts')

# Load data
df = pd.read_csv("sleep.csv")

# for seamless api calling/matching
df = df.rename(columns={
    "Sleep Duration (hours)": "Sleep Duration",
    "Quality of Sleep (scale: 1-10)": "Quality of Sleep",
    "Physical Activity Level (minutes/day)": "Physical Activity Level",
    "Stress Level (scale: 1-10)": "Stress Level",
    "Heart Rate (bpm)": "Heart Rate",
})

# Basic cleanup
df = df.drop(columns=["Person ID"])

# Split BP: "120/80" -> systolic & diastolic
bp = df["Blood Pressure"].str.split("/", expand=True)
df["Systolic"] = bp[0].astype(int)
df["Diastolic"] = bp[1].astype(int)
df = df.drop(columns=["Blood Pressure"])

# Droping rows with NaNs
df = df.dropna(subset=["Sleep Disorder"])

# Define target + features
target_col = "Sleep Disorder"
X = df.drop(columns=[target_col])
y = df[target_col]  # keep as strings: None/Insomnia/Sleep Apnea



# Identify column types
cat_cols = ["Gender", "Occupation", "BMI Category"]
num_cols = [c for c in X.columns if c not in cat_cols]

# Preprocessing 
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# Model 
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

# Split / Train / evaluate 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save a small background dataset for SHAP


X_bg = X_train.sample(100, random_state=42)
joblib.dump(X_bg, ARTIFACTS_DIR / "shap_background.joblib")

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Export pipeline
joblib.dump(pipe, ARTIFACTS_DIR / "sleep_disorder_pipeline.joblib")
print("Saved: sleep_disorder_pipeline.joblib to artifacts directory")
