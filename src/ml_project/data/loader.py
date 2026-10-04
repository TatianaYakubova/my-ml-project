"""Загрузка данных из CSV."""
from pathlib import Path

import pandas as pd

from ml_project.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(path: str | Path) -> pd.DataFrame:
    """Загружает CSV в DataFrame.

    Args:
        path: Путь к CSV-файлу.

    Returns:
        DataFrame с данными.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если DataFrame пустой.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path.resolve()}")

    logger.info("Загрузка данных из %s", path)
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Файл пустой: {path}")

    logger.info("Загружено %d строк, %d колонок", *df.shape)
    return df
