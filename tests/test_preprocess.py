"""Тесты для модуля препроцессинга."""
import numpy as np
import pandas as pd
import pytest

from ml_project.features.preprocess import (
    add_income_category,
    encode_ocean_proximity,
    fill_missing,
    stratified_split,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Тестовый датасет с 10 строками (по 2 на каждую income_cat)."""
    return pd.DataFrame(
        {
            "longitude": [
                -122.23,
                -122.22,
                -122.24,
                -122.25,
                -122.25,
                -118.10,
                -118.20,
                -118.30,
                -118.40,
                -118.50,
            ],
            "latitude": [
                37.88,
                37.86,
                37.85,
                37.85,
                37.85,
                34.10,
                34.20,
                34.30,
                34.40,
                34.50,
            ],
            "housing_median_age": [
                41.0,
                21.0,
                52.0,
                52.0,
                52.0,
                30.0,
                25.0,
                20.0,
                15.0,
                10.0,
            ],
            "total_rooms": [
                880.0,
                7099.0,
                1467.0,
                1274.0,
                1627.0,
                2000.0,
                2200.0,
                2400.0,
                2600.0,
                2800.0,
            ],
            "total_bedrooms": [
                129.0,
                np.nan,
                190.0,
                235.0,
                280.0,
                400.0,
                420.0,
                440.0,
                460.0,
                480.0,
            ],
            "population": [
                322.0,
                2401.0,
                496.0,
                558.0,
                565.0,
                800.0,
                850.0,
                900.0,
                950.0,
                1000.0,
            ],
            "households": [
                126.0,
                1138.0,
                177.0,
                219.0,
                259.0,
                350.0,
                370.0,
                390.0,
                410.0,
                430.0,
            ],
            # 2 значения в каждом income_cat: 1.5, 2.0, 3.0, 4.0, 5.0
            # 2 значения в каждой корзине income_cat (1..5)
            "median_income": [1.2, 1.4, 2.0, 2.8, 3.2, 4.0, 4.8, 5.5, 6.5, 7.0],
            "median_house_value": [
                150000.0,
                160000.0,
                200000.0,
                210000.0,
                250000.0,
                260000.0,
                300000.0,
                310000.0,
                350000.0,
                360000.0,
            ],
            "ocean_proximity": ["NEAR BAY"] * 5 + ["INLAND"] * 5,
        }
    )


def test_add_income_category(sample_df: pd.DataFrame) -> None:
    """income_cat создаётся и не превышает 5."""
    result = add_income_category(sample_df)
    assert "income_cat" in result.columns
    assert result["income_cat"].max() <= 5.0


def test_fill_missing(sample_df: pd.DataFrame) -> None:
    """Пропуски в total_bedrooms заполняются."""
    result = fill_missing(sample_df)
    assert result["total_bedrooms"].isna().sum() == 0


def test_encode_ocean_proximity(sample_df: pd.DataFrame) -> None:
    """ocean_proximity заменяется на One-Hot колонки."""
    train, test = encode_ocean_proximity(sample_df.copy(), sample_df.copy())
    assert "ocean_proximity" not in train.columns
    ocean_cols = [c for c in train.columns if c.startswith("ocean_proximity_")]
    assert len(ocean_cols) > 0


def test_stratified_split(sample_df: pd.DataFrame) -> None:
    """Split не теряет строки и возвращает 2 набора."""
    df_with_cat = add_income_category(sample_df)
    train, test = stratified_split(df_with_cat, test_size=0.5)
    assert len(train) + len(test) == len(sample_df)
    assert "income_cat" not in train.columns
