from pathlib import Path
import joblib
import mlflow
import mlflow.catboost

MODEL_URI = "models:/real_estate_price_model/4"  # или /Production
OUT_PATH = Path("services/models/Sprint2_cb.pkl")

model = mlflow.catboost.load_model(MODEL_URI)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, OUT_PATH)

print("Saved:", OUT_PATH)
print("Type:", type(model))
