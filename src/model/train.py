import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from pathlib import Path
from configs.paths import DATA_DIR, ARTIFACTS_DIR, CONFIG_DIR
from omegaconf import OmegaConf

class ModelTrainer:
    def __init__(self, experiment_name: str = "model_training"):
        self.config = OmegaConf.load(CONFIG_DIR / "model_params.yaml")
        self.train_parquet_path = DATA_DIR / "processed" / "train.parquet"
        self.test_parquet_path = DATA_DIR / "processed" / "test.parquet"
        self.model = None
        self.metrics = {}
        self.n_features_ = None
        
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)
    
    def load_train_data(self):
        return pd.read_parquet(self.train_parquet_path)

    def load_test_data(self):
        return pd.read_parquet(self.test_parquet_path)

    def _compute_metrics(self, y_true, y_pred, n_features):
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        n = len(y_true)
        y_mean = y_true.mean()
        
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_true - y_mean)**2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        
        metrics = {
            "scope": y_true.max() - y_true.min(),
            "rmse": np.sqrt(np.mean(errors**2)),
            "mae": np.mean(abs_errors),
            "mape": np.mean(np.abs(errors / y_true)),  # доля (соответствует исходному коду)
            "r2": r2,
            "adjusted_r2": 1 - ((1 - r2) * (n - 1) / (n - n_features - 1)) if n > n_features + 1 else np.nan,
            "bias": errors.mean(),
            "bias_percent": errors.mean() / y_mean if y_mean != 0 else np.nan,
            "max_error": abs_errors.max()
        }
        return metrics

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise RuntimeError("Модель ещё не обучена. Сначала вызовите train().")
        
        y_pred = self.model.predict(X_test)
        self.metrics = self._compute_metrics(y_test, y_pred, self.n_features_)

    def train(self):
        train_df = self.load_train_data()
        test_df = self.load_test_data()

        X_train = train_df.drop("SalePrice", axis=1)
        y_train = train_df.SalePrice
        X_test = test_df.drop("SalePrice", axis=1)
        y_test = test_df.SalePrice

        self.n_features_ = X_train.shape[1]

        self.model = xgb.XGBRegressor(**self.config.training)
        self.model.fit(X_train, y_train)

        self.evaluate(X_test, y_test)

    def save(self):
        with mlflow.start_run():
            mlflow.log_params(self.config.training)
            mlflow.log_metrics(self.metrics)

            mlflow.xgboost.log_model(self.model, self.config.model.type)

            model_path = ARTIFACTS_DIR / "models/model.json"
            self.model.save_model(model_path)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
    trainer.save()