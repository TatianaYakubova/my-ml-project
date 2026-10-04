"""Загрузка конфигурации из YAML."""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Читает YAML-конфиг и возвращает словарь.

    Args:
        path: Путь к файлу конфигурации.

    Returns:
        Словарь с настройками.

    Raises:
        FileNotFoundError: Если файл не найден.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {config_path.resolve()}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
