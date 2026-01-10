# Sleep Disorder Predictor (RandomForest + SHAP + LLM)

A portfolio project that predicts sleep disorder category (**None**, **Insomnia**, **Sleep Apnea**) using a scikit-learn pipeline.
It also provides an explanation using **SHAP** (model-grounded factors) and an **LLM** (human-friendly summary).

> Educational only — not medical advice.

## Features
- FastAPI inference API
- `/predict` (fast prediction)
- `/predict_with_explanation` (prediction + SHAP factors + LLM explanation)
- Simple HTML frontend (no framework)

## Tech Stack
- Python, scikit-learn, pandas, numpy
- FastAPI + Uvicorn
- SHAP
- OpenAI SDK (LLM explanation)

## Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
