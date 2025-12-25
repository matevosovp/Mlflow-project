# Улучшение baseline-модели с MLflow (Яндекс Недвижимость)

Проект демонстрирует воспроизводимый цикл экспериментов MLflow для регрессии стоимости недвижимости.
Источник данных: PostgreSQL таблица `public.real_estate_dataset_clean`, целевая переменная `price`.

Требование: все эксперименты ведутся в одном MLflow experiment `model_FE_matevosov`.
Модели должны быть зарегистрированы в MLflow Model Registry минимум в 4 версиях:
1) baseline из S3
2) модель после генерации признаков
3) модель после отбора признаков
4) модель после подбора гиперпараметров

## Технологии

- MLflow Tracking Server + Model Registry
- PostgreSQL как backend store и registry store
- Yandex Object Storage (S3) как artifact store
- sklearn, catboost, optuna, mlxtend, autofeat

## Бакет для артефактов

`s3-student-mle-20251010-5a382f9c3d`

## Структура репозитория

- `mlflow_server/`
  - `start_mlflow.sh` запуск MLflow server (tracking + registry)
  - `register_baseline.py` регистрация baseline модели из S3
- `model_improvement/`
  - `model_improvement.ipynb` один ноутбук со всеми этапами 2–5

## Переменные окружения

Создайте файл `.env` в корне репозитория (не коммитится).
Минимально необходимо:

- PostgreSQL (backend store и registry store)
  - `DB_DESTINATION_HOST`
  - `DB_DESTINATION_PORT`
  - `DB_DESTINATION_USER`
  - `DB_DESTINATION_PASSWORD`
  - `DB_DESTINATION_DBNAME`

- S3 (Yandex Object Storage)
  - `MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net`
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`

- MLflow
  - `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`

## Установка зависимостей

### Вариант Bash (Linux, WSL, macOS)

```bash

# обновление локального индекса пакетов
sudo apt-get update
# установка расширения для виртуального пространства
sudo apt-get install python3.10-venv
# создание виртуального пространства
python3.10 -m venv .venv_project_name

source .venv_project_name/bin/activate

```
Заполните .env_template и переименуйте в .env

```bash
# экспортируйте перепенные из .env
export $(grep -v '^#' .env | xargs)
```



