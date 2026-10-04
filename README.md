# My ML Project

Production-ready ML pipeline для предсказания стоимости жилья в Калифорнии.

## Стек
Python 3.13, Poetry, scikit-learn, pandas, numpy, pre-commit (black, ruff, isort), pytest.

## Установка
poetry install

## Обучение
poetry run python -m ml_project.models.train

Метрики: MAE ≈ 35 617, RMSE ≈ 52 040.

## Тесты
poetry run pytest -v

## Проверка кодстайла
poetry run pre-commit run --all-files

## Структура
src/ml_project/data/loader.py
src/ml_project/features/preprocess.py
src/ml_project/models/train.py
src/ml_project/utils/{config,logger}.py
