# Heart Disease Detection

![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=flat&logo=kaggle&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=flat&logo=numpy&logoColor=blue)

Проект по дисциплине "Введение в науки о данных", заключающийся в построении ML пайплайна для решения задачи бинарной классификации, в частности для определения наличия сердечных заболеваний.

Ссылка на датасет: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

# Содержание
- [Технологии](#Технологии)
- [Файловая структура](#файловая-структура)
- [Разработка](#Разработка)
- [Использование](#Использование)

# Технологии
- Язык программирования: [Python](https://www.python.org/)
- Линтер / форматтер: [Ruff](https://github.com/astral-sh/ruff)
- Библиотеки:
  - Обработка данных: [pandas](https://pandas.pydata.org/)
  - Визуализация:
    - [matplotlib](https://matplotlib.org/)
    - [seaborn](https://seaborn.pydata.org/)
  - Машинное обучение: [scikit-learn](https://scikit-learn.org/stable/)
- Система контроля версий: [Git](https://git-scm.com/) + [GitHub](https://github.com/)
- MLOps: [MLFlow](https://mlflow.org/)

# Файловая структура
- **`/src`**: Исходный код проекта.
  - `/notebooks`: Sandbox блокноты.
  - `/tests`: Тесты для проекта.
  - `/stages`: Этапы пайплайна.
- **`pyproject.toml`**: Конфигурация инструментов проекта.
- **`requirements.txt`**: Зависимости проекта.

# Разработка

## Требования
Для установки и запуска проекта, необходим Python 3.13.5+ (за работоспособность на версиях ниже ответственности не несём).

## Создание виртуальной среды
```sh
py -m venv .venv
```

## Активация виртуальной среды

### Windows PowerShell
```sh
./.venv/Scripts/Activate.ps1
```

### CMD
```sh
./.venv/Scripts/activate.bat
```

## Установка зависимостей

```sh
pip install -r requirements.txt
```

# Тестирование
TODO

# Использование
TODO

# Команда проекта
- Data Scientist - [Косов Тимофей Николаевич](https://github.com/tuzikisreal)
- DevOps Engineer - [Бойко Георгий Алексеевич](https://github.com/ihyall)
