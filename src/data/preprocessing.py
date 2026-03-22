import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, TargetEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from configs.paths import DATA_DIR, ARTIFACTS_DIR


class DataPreprocessor:
    def __init__(self, target_col: str = "SalePrice", feature_config: Optional[Dict] = None, experiment_name: str = "preprocessing"):
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_config: Dict[str, List[str]] = feature_config or {}
        self.target_col = target_col
        mlflow.set_experiment(experiment_name)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "MSSubClass" in df.columns:
            df["MSSubClass"] = df["MSSubClass"].astype(str)
        if "Id" in df.columns:
            df.drop("Id", axis=1, inplace=True)
        return df

    def _define_features(self, df: pd.DataFrame) -> None:
        ready_quality = ["OverallQual", "OverallCond"]

        numeric = df.select_dtypes(include=[np.number]).drop(columns=ready_quality, errors='ignore').columns.tolist()
        categorical = df.drop(columns=numeric + ready_quality, errors='ignore').columns.tolist()

        ordinal_list = [
            "OverallQual", "OverallCond",
            "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
            "BsmtExposure", "HeatingQC", "KitchenQual",
            "FireplaceQu", "GarageQual", "GarageCond", "PoolQC", "Fence"
        ]

        high_card = [c for c in categorical if c not in ordinal_list]
        nunique = df[high_card].nunique()
        nominal = nunique[nunique <= 10].index.tolist()
        high_cardinality = [c for c in high_card if c not in nominal]

        if self.target_col in df.columns:
            for group in [numeric, nominal, ordinal_list, high_cardinality]:
                if self.target_col in group:
                    group.remove(self.target_col)

        self.feature_config = {
            "numeric": numeric,
            "nominal": nominal,
            "ordinal": ordinal_list,
            "high_cardinality": high_cardinality,
            "target": self.target_col if self.target_col in df.columns else None
        }

    def set_feature_config_from_df(self, df: pd.DataFrame) -> None:
        df_clean = self._prepare_data(df)
        self._define_features(df_clean)

    def _create_pipelines(self) -> ColumnTransformer:
        pipes = {
            "num": make_pipeline(
                SimpleImputer(strategy="median"),
                RobustScaler()
            ),
            "ohe": make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ),
            "ord": make_pipeline(
                SimpleImputer(strategy="constant", fill_value="Missing"),
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1
                )
            ),
            "tar": make_pipeline(
                SimpleImputer(strategy="constant", fill_value="Missing"),
                TargetEncoder(smooth="auto", target_type='continuous'),
                RobustScaler()
            )
        }

        return ColumnTransformer(
            transformers=[
                ("num", pipes["num"], self.feature_config["numeric"]),
                ("ohe", pipes["ohe"], self.feature_config["nominal"]),
                ("ord", pipes["ord"], self.feature_config["ordinal"]),
                ("tar", pipes["tar"], self.feature_config["high_cardinality"])
            ],
            remainder="drop"
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        X_clean = self._prepare_data(X)

        if not self.feature_config:
            if y is not None:
                df_temp = X_clean.copy()
                df_temp[self.target_col] = y
                self._define_features(df_temp)
            else:
                self._define_features(X_clean)

        self.preprocessor = self._create_pipelines()
        self.preprocessor.set_output(transform="pandas")
        self.preprocessor.fit(X_clean, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() or load() first.")
        X_clean = self._prepare_data(X)
        return self.preprocessor.transform(X_clean)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def save(self, name: str, use_mlflow: bool = True, compression: int = 0) -> None:
        path = ARTIFACTS_DIR / f"preprocessors/{name}"
        joblib.dump(self.preprocessor, path, compression)

        if use_mlflow:
            with mlflow.start_run():
                mlflow.sklearn.log_model(self.preprocessor, "preprocessor")
                mlflow.log_params({"features": str(self.feature_config)})
                mlflow.log_artifact(path)

    def load(self, name: str) -> None:
        path = ARTIFACTS_DIR / f"preprocessors/{name}"
        self.preprocessor = joblib.load(path)


if __name__ == "__main__":
    prep = DataPreprocessor()

    data = pd.read_csv(DATA_DIR / "raw/train.csv")
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    prep.fit(X_train, y_train)

    processed_train = prep.transform(X_train)
    processed_test = prep.transform(X_test)

    processed_train[prep.target_col] = y_train.values
    processed_test[prep.target_col] = y_test.values

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_train.to_parquet(processed_dir / "train.parquet", index=False)
    processed_test.to_parquet(processed_dir / "test.parquet", index=False)

    prep.save("preprocessor.joblib", compression=3)