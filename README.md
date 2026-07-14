# MLflow: улучшение и регистрация модели недвижимости

[![Quality checks](https://github.com/matevosovp/Mlflow-project/actions/workflows/quality.yml/badge.svg)](https://github.com/matevosovp/Mlflow-project/actions/workflows/quality.yml)

Проект демонстрирует воспроизводимый цикл экспериментов для регрессии стоимости недвижимости: baseline-модель импортируется из S3, последовательно улучшается и регистрируется в MLflow Model Registry.

## Коротко о проекте

- PostgreSQL используется как источник данных, backend store и Model Registry store;
- S3-compatible storage хранит модели и experiment artifacts;
- все стадии сравниваются внутри одного MLflow experiment;
- параметры, метрики, окружение и lineage модели сохраняются в runs;
- в Registry создаются отдельные версии baseline, feature engineering, feature selection и tuned-модели;
- конфигурация вынесена в переменные окружения, без личных bucket names и секретов в коде.

## Архитектура

```text
PostgreSQL: public.real_estate_dataset_clean
                    │
                    ▼
          model_improvement.ipynb
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
MLflow Tracking            MLflow Artifacts
PostgreSQL backend         S3-compatible storage
        │                        │
        └───────────┬────────────┘
                    ▼
             Model Registry
        baseline → FE → selection → tuning
```

## Этапы эксперимента

| Этап | Что происходит | Результат в MLflow |
|---|---|---|
| 1. Baseline | загрузка готовой модели из S3 и оценка на едином holdout | первая зарегистрированная версия |
| 2. Feature engineering | генерация и проверка новых признаков | отдельный run и версия модели |
| 3. Feature selection | отбор полезных признаков | сравнимые метрики и artifacts |
| 4. Hyperparameter tuning | подбор параметров CatBoost через Optuna | финальная tuned-версия |

Основные метрики регрессии: **RMSE, MAE и R²**. Фиксированный `random_state=42` и единый test split позволяют корректно сравнивать стадии между собой.

## Что логируется

- параметры данных и модели;
- RMSE, MAE и R² на holdout;
- размер датасета и число признаков;
- исходная и MLflow-упакованная модели;
- input example и model signature;
- `pip freeze` для восстановления окружения;
- теги stage, lineage, source и data table;
- описание и метрики каждой версии в Model Registry.

Скрипт [`register_baseline.py`](mlflow_server/register_baseline.py) связывает исходную baseline-модель, данные, run и созданную версию Registry.

## Структура репозитория

```text
.
├── mlflow_server/
│   ├── start_mlflow.sh          # запуск Tracking Server и Registry
│   └── register_baseline.py     # импорт и регистрация baseline
├── model_improvement/
│   └── model_improvement.ipynb  # feature engineering, selection и tuning
├── requirements.txt
└── .env.example
```

## Быстрый старт

Требуются Python 3.10, доступ к PostgreSQL и S3-compatible storage.

```bash
git clone https://github.com/matevosovp/Mlflow-project.git
cd Mlflow-project

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# заполните PostgreSQL, S3 и BASELINE_MODEL_S3_URI
```

Реальные секреты должны находиться только в локальном `.env`; этот файл исключён из Git.

### 1. Запустить MLflow

```bash
bash mlflow_server/start_mlflow.sh
```

По умолчанию интерфейс будет доступен на `http://127.0.0.1:5000`. Host и port настраиваются через `MLFLOW_HOST` и `MLFLOW_PORT`.

Startup script:

- проверяет наличие обязательных переменных;
- безопасно URL-кодирует имя пользователя и пароль PostgreSQL;
- не выводит пароль или полный database URI в лог;
- использует `S3_BUCKET_NAME` только из окружения.

### 2. Зарегистрировать baseline

```bash
python mlflow_server/register_baseline.py
```

Путь к исходной модели передаётся через `BASELINE_MODEL_S3_URI`. Названия experiment и registered model можно переопределить переменными `MLFLOW_EXPERIMENT_NAME` и `MLFLOW_REGISTERED_MODEL_NAME`.

### 3. Запустить эксперименты

```bash
jupyter lab model_improvement/model_improvement.ipynb
```

Выполняйте ноутбук последовательно после успешной регистрации baseline, чтобы все версии модели попали в один experiment и Registry.

## Переменные окружения

| Группа | Переменные |
|---|---|
| PostgreSQL | `DB_DESTINATION_HOST`, `DB_DESTINATION_PORT`, `DB_DESTINATION_NAME`, `DB_DESTINATION_USER`, `DB_DESTINATION_PASSWORD` |
| S3 | `MLFLOW_S3_ENDPOINT_URL`, `S3_BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `BASELINE_MODEL_S3_URI` |
| MLflow | `MLFLOW_TRACKING_URI`, `MLFLOW_HOST`, `MLFLOW_PORT`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_REGISTERED_MODEL_NAME` |

Полный безопасный шаблон находится в [`.env.example`](.env.example).

## Проверка качества

GitHub Actions автоматически проверяет:

- синтаксис Python-модулей;
- корректность startup shell script;
- целостность JSON-структуры ноутбука.

Локальный запуск тех же базовых проверок:

```bash
python -m compileall -q mlflow_server
bash -n mlflow_server/start_mlflow.sh
python -m json.tool model_improvement/model_improvement.ipynb > /dev/null
```

## Ограничения

- PostgreSQL и S3 предоставляются отдельно и не поднимаются этим репозиторием;
- основной experiment workflow оформлен в ноутбуке, а не в отдельном CLI-пайплайне;
- проект отвечает за tracking и registry, но не за online serving;
- для production следует добавить аутентификацию и TLS перед публикацией MLflow UI.

Следующий этап жизненного цикла модели — [FastAPI-сервис с Prometheus и Grafana](https://github.com/matevosovp/ML-model_deployment_in_a_cloud_infrastructure).
