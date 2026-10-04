"""Препроцессинг данных California Housing."""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ml_project.utils.logger import get_logger

logger = get_logger(__name__)


def add_income_category(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку income_cat для стратификации.

    Args:
        df: Исходный DataFrame.

    Returns:
        DataFrame с добавленной колонкой income_cat.
    """
    df = df.copy()
    df["income_cat"] = np.ceil(df["median_income"] / 1.5)
    df["income_cat"] = df["income_cat"].where(df["income_cat"] < 5, 5.0)
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит данные на train/test по стратам income_cat.

    Args:
        df: DataFrame с колонкой income_cat.
        test_size: Доля теста.
        random_state: Seed.

    Returns:
        Кортеж (train_set, test_set) без колонки income_cat.
    """
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    for train_idx, test_idx in split.split(df, df["income_cat"]):
        strat_train = df.loc[train_idx].drop("income_cat", axis=1)
        strat_test = df.loc[test_idx].drop("income_cat", axis=1)

    logger.info(
        "Стратифицированный split: train=%d, test=%d",
        len(strat_train),
        len(strat_test),
    )
    return strat_train, strat_test


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски в total_bedrooms медианой.

    Args:
        df: DataFrame.

    Returns:
        DataFrame без пропусков.
    """
    df = df.copy()
    median_bedrooms = df["total_bedrooms"].median()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(median_bedrooms)
    logger.info("Пропуски в total_bedrooms заполнены медианой: %.1f", median_bedrooms)
    return df


def encode_ocean_proximity(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-Hot-кодирование категориальной колонки ocean_proximity.

    Args:
        train: Обучающая выборка.
        test: Тестовая выборка.

    Returns:
        Кортеж (train_encoded, test_encoded).
    """
    train = pd.get_dummies(train, columns=["ocean_proximity"], drop_first=False)
    test = pd.get_dummies(test, columns=["ocean_proximity"], drop_first=False)

    # Синхронизируем колонки: некоторые категории могут быть только в train
    test = test.reindex(columns=train.columns, fill_value=0)

    logger.info("Колонки после One-Hot: %d", train.shape[1])
    return train, test


def preprocess(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Полный пайплайн препроцессинга.

    Args:
        df: Исходный DataFrame.
        config: Словарь с настройками (data.test_size, data.random_state).

    Returns:
        Кортеж (train_processed, test_processed).
    """
    df = add_income_category(df)

    train, test = stratified_split(
        df,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    train = fill_missing(train)
    test = fill_missing(test)

    train, test = encode_ocean_proximity(train, test)

    logger.info("Препроцессинг завершён. Train: %s, Test: %s", train.shape, test.shape)
    return train, test
