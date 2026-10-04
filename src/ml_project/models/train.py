"""Обучение модели RandomForestRegressor."""
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from ml_project.data.loader import load_data
from ml_project.features.preprocess import preprocess
from ml_project.utils.config import load_config
from ml_project.utils.logger import get_logger

logger = get_logger(__name__)


def split_features_target(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Отделяет признаки от целевой переменной."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> RandomForestRegressor:
    """Обучает RandomForestRegressor и возвращает модель."""
    logger.info(
        "Обучение RandomForest: n_estimators=%d, max_depth=%d",
        n_estimators,
        max_depth,
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Считает MAE и RMSE на тесте."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    logger.info("MAE = %.2f, RMSE = %.2f", mae, rmse)
    return {"mae": mae, "rmse": rmse}


def save_model(model: RandomForestRegressor, path: str | Path) -> None:
    """Сохраняет модель на диск."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Модель сохранена: %s", path)


def run(config_path: str = "configs/config.yaml") -> None:
    """Полный пайплайн: загрузка → препроцессинг → обучение → сохранение."""
    config: dict[str, Any] = load_config(config_path)

    df = load_data(config["data"]["path"])
    train_df, test_df = preprocess(df, config)

    target = config["data"]["target"]
    X_train, y_train = split_features_target(train_df, target)
    X_test, y_test = split_features_target(test_df, target)

    model = train_model(
        X_train,
        y_train,
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"],
    )

    metrics = evaluate(model, X_test, y_test)
    logger.info("Метрики: %s", metrics)

    save_model(model, config["model"]["output_path"])


if __name__ == "__main__":
    run()
