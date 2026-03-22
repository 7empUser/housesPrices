import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from configs.paths import DATA_DIR, ARTIFACTS_DIR

def get_prediction(data: dict) -> float:
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessors/preprocessor.joblib")
    df = pd.DataFrame([data])
    df = preprocessor.transform(df)

    model = xgb.XGBRegressor()
    model.load_model(ARTIFACTS_DIR / "models/model.json")
    prediction = model.predict(df)[0]
    return float(prediction)