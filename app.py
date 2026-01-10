from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import joblib
import numpy as np
import os
import pandas as pd
import shap
from openai import OpenAI
import traceback
from pathlib import Path



# App + CORS
app = FastAPI(title="Sleep Disorder Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # For demo. In production, restrict to your frontend domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Safe OpenAI client init
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None



# Load exported pipeline + SHAP background
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

pipe = joblib.load(ARTIFACTS_DIR / "sleep_disorder_pipeline.joblib")
X_bg = joblib.load(ARTIFACTS_DIR / "shap_background.joblib")

# Verify input mapping (This can be optional)
print("Expected columns:", getattr(pipe, "feature_names_in_", "N/A"))

pre = pipe.named_steps["preprocess"]
rf = pipe.named_steps["model"]
feature_names = pre.get_feature_names_out()

# SHAP background must be in transformed space
X_bg_transformed = pre.transform(X_bg)

# TreeExplainer works well for RandomForest-like models
explainer = shap.TreeExplainer(
    rf,
    X_bg_transformed,
    feature_names=feature_names
)



# Pydantic request schema
class SleepInput(BaseModel):
    Gender: str = Field(..., examples=["Male", "Female"])
    Age: int = Field(..., ge=1, le=120)
    Occupation: str = Field(..., examples=["Engineer", "Doctor", "Teacher"])
    Sleep_Duration_hours: float = Field(..., ge=0, le=24)
    Quality_of_Sleep: int = Field(..., ge=1, le=10)
    Physical_Activity_Level_minutes_day: int = Field(..., ge=0, le=10000)
    Stress_Level: int = Field(..., ge=1, le=10)
    BMI_Category: str = Field(..., examples=["Underweight", "Normal", "Overweight"])
    Heart_Rate_bpm: int = Field(..., ge=20, le=250)
    Daily_Steps: int = Field(..., ge=0, le=100000)
    Systolic: int = Field(..., ge=50, le=250)
    Diastolic: int = Field(..., ge=30, le=200)


def to_model_df(x: SleepInput) -> pd.DataFrame:
    """
    Convert API input -> DataFrame with the EXACT training column names.

    IMPORTANT:
    These keys MUST match the columns used in training.
    Based on your earlier error, the pipeline expects:
    Sleep Duration, Quality of Sleep, Physical Activity Level, Stress Level, Heart Rate, etc.
    """
    return pd.DataFrame([{
        "Gender": x.Gender,
        "Age": x.Age,
        "Occupation": x.Occupation,

        "Sleep Duration": x.Sleep_Duration_hours,
        "Quality of Sleep": x.Quality_of_Sleep,
        "Physical Activity Level": x.Physical_Activity_Level_minutes_day,
        "Stress Level": x.Stress_Level,
        "Heart Rate": x.Heart_Rate_bpm,

        "Daily Steps": x.Daily_Steps,
        "BMI Category": x.BMI_Category,

        "Systolic": x.Systolic,
        "Diastolic": x.Diastolic,
    }])


def pretty_feature_name(raw: str) -> str:
    # Examples: "num__Age" -> "Age", "num__Daily Steps" -> "Daily Steps"
    #           "cat__Occupation_Nurse" -> "Occupation (Nurse)"
    if raw.startswith("num__"):
        return raw.replace("num__", "")
    if raw.startswith("cat__"):
        x = raw.replace("cat__", "")
        # try to split one-hot "Column_Value"
        if "_" in x:
            col, val = x.split("_", 1)
            return f"{col} ({val})"
        return x
    return raw


# SHAP helper
def shap_top_reasons(input_df: pd.DataFrame, top_k: int = 5):
    x_t = pre.transform(input_df)

    proba_arr = rf.predict_proba(x_t)[0]
    pred_idx = int(np.argmax(proba_arr))
    pred_label = str(rf.classes_[pred_idx])

    shap_vals = explainer.shap_values(x_t)

    # Robust extraction of a single vector of SHAP values (n_features,)
    if isinstance(shap_vals, list):
        # list[class] -> (n_samples, n_features)
        sv = shap_vals[pred_idx][0]
    else:
        # Could be (n_samples, n_features) OR (n_samples, n_features, n_classes)
        sv = shap_vals
        if sv.ndim == 3:
            sv = sv[0, :, pred_idx]
        else:
            sv = sv[0]

    impacts = np.abs(sv)
    top_idx = impacts.argsort()[::-1][:top_k]

    reasons = []
    for i in top_idx:
        direction = "increased" if sv[i] > 0 else "decreased"
        reasons.append({
            "feature": pretty_feature_name(str(feature_names[i])),
            "raw_feature": str(feature_names[i]),
            "direction": direction,
            "shap_value": float(sv[i]),
            "impact": float(impacts[i])
        })

    probabilities = {str(c): float(p) for c, p in zip(rf.classes_, proba_arr)}
    return pred_label, probabilities, reasons


# -----------------------------
# 5) LLM explanation helper
# -----------------------------
def llm_explain(pred_label: str, probabilities: dict, input_payload: dict, shap_reasons: list) -> str:
    """
    Uses an LLM to explain the SHAP factors in plain language.
    Guardrails:
      - no diagnosis
      - only use provided inputs + SHAP factors
      - include disclaimer
    """
    if client is None:
        return "OPENAI_API_KEY is not set. LLM explanation is disabled."

    prompt = f"""
    Write a short, very simple explanation that a beginner can understand.

    Rules:
    - Do NOT mention feature codes like "num__" or "cat__".
    - Do NOT mention SHAP values or decimals.
    - Do NOT say weird things like "since you're not a nurse..."
    - Use the user's actual values.
    - Explain in everyday language what likely pushed the model toward the predicted class.
    - Keep it short.

    Return exactly:
    1) One sentence: "The model predicted X."
    2) "Top reasons" as 3 bullets (simple).
    3) "What you can do" as 3 bullets (general, not medical).
    4) One short disclaimer line.

    Prediction: {pred_label}
    User inputs: {input_payload}
    Top factors (already cleaned): {shap_reasons}
    """.strip()

    resp = client.responses.create(
        model="gpt-5.2",
        input=prompt
    )
    return resp.output_text


# -----------------------------
# 6) Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/routes")
def routes():
    return [{"path": r.path, "methods": sorted(list(r.methods))} for r in app.routes]


@app.post("/predict")
def predict(inp: SleepInput):
    """
    Fast prediction endpoint (no SHAP/LLM).
    """
    try:
        df = to_model_df(inp)
        pred = pipe.predict(df)[0]

        proba = None
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(df)[0]
            classes = pipe.named_steps["model"].classes_
            proba = {str(cls): float(p) for cls, p in zip(classes, probs)}

        return {"prediction": str(pred), "probabilities": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_with_explanation")
def predict_with_explanation(inp: SleepInput):
    try:
        df = to_model_df(inp)

        pred_label, proba, reasons = shap_top_reasons(df, top_k=5)

        explanation = llm_explain(
            pred_label=pred_label,
            probabilities=proba,
            input_payload=inp.model_dump(),
            shap_reasons=reasons
        )

        return {
            "prediction": pred_label,
            "probabilities": proba,
            "top_shap_factors": reasons,
            "explanation": explanation
        }

    except Exception as e:
        print("ERROR in /predict_with_explanation:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    
