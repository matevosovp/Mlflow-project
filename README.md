# MLflow: воспроизводимое улучшение модели недвижимости

[![Quality checks](https://github.com/matevosovp/Mlflow-project/actions/workflows/quality.yml/badge.svg)](https://github.com/matevosovp/Mlflow-project/actions/workflows/quality.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

Production-oriented учебный проект, который показывает полный цикл обучения регрессионной модели: безопасное извлечение данных из PostgreSQL, воспроизводимый train/validation/test workflow, подбор CatBoost, логирование в MLflow и регистрацию готового raw-input pipeline в Model Registry.

Главный акцент проекта — не только получить метрику, но и сделать результат проверяемым, воспроизводимым и пригодным для следующего шага: online serving.

Ограничения, intended use и риски описаны отдельно в [`MODEL_CARD.md`](MODEL_CARD.md); правила безопасной работы с секретами и model artifacts — в [`SECURITY.md`](SECURITY.md).

## Что демонстрирует проект

- PostgreSQL как источник данных, MLflow backend и Model Registry store;
- S3-compatible storage для моделей и experiment artifacts;
- безопасный SQLAlchemy Core query с явным контрактом из 18 колонок;
- group-aware выборки примерно 60% train, 20% validation, 20% test без пересечения зданий;
- Optuna и `RandomizedSearchCV`, оцениваемые на одной отдельной validation-выборке;
- единый сериализуемый sklearn Pipeline от сырых признаков до CatBoost prediction;
- dataset fingerprint, model signature, input example, параметры и метрики в MLflow;
- привязка Registry version к точному `run_id`, а не к «последней» версии;
- автоматические unit tests, lint, dependency check и notebook validation в CI.

## Архитектура

```text
PostgreSQL
    │  SQLAlchemy reflection + explicit column contract
    ▼
Validated DataFrame
    │
    ├── train ≈60% ─────── fit / tuning
    ├── validation ≈20% ── model selection only
    └── test ≈20% ──────── one final evaluation
                              │
Raw input ─► domain features ─► imputation / one-hot ─► selection ─► CatBoost
                              │
                              ▼
                     MLflow Tracking + S3
                              │
                              ▼
                Model Registry alias: candidate
```

Разбиение выполняется по `building_id`: квартиры одного здания не могут оказаться в разных частях. Test labels доступны только в финальной части `train_and_register`: после выбора параметров модель один раз переобучается на train+validation и один раз оценивается на test.

## Model contract

Зарегистрированная модель принимает исходные бизнес-признаки таблицы, а не заранее подготовленную матрицу. Технические идентификаторы не входят в serving contract. В MLflow сохраняется весь pipeline:

1. удаление идентификаторов `flat_id` и `building_id` до формирования signature;
2. детерминированные domain features: возраст здания, доли площадей, положение этажа и другие отношения;
3. imputation и обработка неизвестных категорий;
4. feature selection внутри Pipeline;
5. sklearn-совместимый адаптер CatBoost;
6. input example и signature для проверки входной схемы.

Это позволяет загрузить Registry version и вызвать `predict` непосредственно на исходном DataFrame.

## Структура

```text
.
├── mlflow_project/
│   ├── config.py       # typed env configuration и безопасный database URL
│   ├── data.py         # SQL extraction, data contract, fingerprint и split
│   ├── features.py     # raw-input feature/model Pipeline
│   ├── training.py     # tuning, final test, MLflow logging и registration
│   └── registry.py     # выбор версии строго по run_id
├── mlflow_server/
│   ├── start_mlflow.sh
│   └── register_baseline.py
├── model_improvement/
│   └── model_improvement.ipynb  # тонкий review-friendly entry point
├── tests/
├── requirements.txt
├── requirements-notebook.txt
├── requirements-dev.txt
└── .env.example
```

Логика намеренно вынесена из notebook в Python package. Notebook показывает этапы и вызывает тестируемые функции, поэтому код обучения не дублируется в нескольких местах.

## Быстрый старт

Требуются Python 3.12, PostgreSQL и S3-compatible storage.

```bash
git clone https://github.com/matevosovp/Mlflow-project.git
cd Mlflow-project

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-notebook.txt

cp .env.example .env
# заполните подключения и параметры baseline
```

Python entry points и startup script автоматически читают корневой `.env`. Пароль PostgreSQL передаётся через `sqlalchemy.URL`, поэтому специальные символы корректно кодируются и не выводятся в лог.

### 1. Запустить MLflow

```bash
bash mlflow_server/start_mlflow.sh
```

По умолчанию UI доступен на `http://127.0.0.1:5000`. Tracking Server использует PostgreSQL для backend/Registry и S3 bucket как artifact root. С `--no-serve-artifacts` MLflow clients обращаются к S3 напрямую и должны иметь собственные AWS/IAM credentials.

### 2. Зарегистрировать проверенный baseline

Сначала вычислите SHA-256 доверенного model artifact и укажите его в `BASELINE_MODEL_SHA256`. Скрипт откажется десериализовать файл, если digest не совпадает.

```bash
python -m mlflow_server.register_baseline
```

Baseline оценивается на том же детерминированном test split и получает Registry alias `baseline`.

### 3. Обучить и зарегистрировать candidate

```bash
python -m mlflow_project.training
```

Можно также открыть демонстрационный notebook:

```bash
jupyter lab model_improvement/model_improvement.ipynb
```

Финальная версия получает alias `candidate`. Продвижение в `champion` намеренно не автоматизировано: это отдельное решение после сравнения с baseline и проверки бизнес-порога.

## Что сохраняется в MLflow

- полный конфиг split и random seed;
- SHA-256 fingerprint датасета;
- результаты Optuna и Randomized Search на validation;
- единственные финальные RMSE, MAE и R² на test;
- выбранный tuning method и параметры;
- raw input example и model signature;
- полный preprocessing + model Pipeline;
- Registry tags, описание, run lineage и alias.

Старые локальные графики и метрики не хранятся в Git: они легко устаревают и могут расходиться с Registry. Каждый воспроизводимый запуск сохраняет отчёты рядом с моделью в MLflow artifacts.

## Проверка качества

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
python -m compileall -q mlflow_project mlflow_server
bash -n mlflow_server/start_mlflow.sh
python -m json.tool model_improvement/model_improvement.ipynb > /dev/null
```

Тесты проверяют:

- отсутствие пересечений строк и `building_id` между train/validation/test;
- правильное использование `PredefinedSplit`;
- блокировку небезопасных SQL identifiers;
- URL-кодирование credentials;
- raw-input prediction полного Pipeline;
- выбор Registry version по текущему `run_id`.

GitHub Actions запускает эти проверки на каждом push и pull request.

## Безопасность и ограничения

- `.env`, данные, модели и локальные artifacts исключены из Git;
- baseline deserialization разрешается только после проверки доверенного SHA-256;
- S3 credentials могут поставляться стандартной boto3 credential chain или IAM role;
- MLflow UI запускается на loopback interface; для внешнего доступа нужны TLS, authentication и reverse proxy;
- PostgreSQL и S3 являются внешней инфраструктурой и не создаются этим репозиторием;
- online serving находится в следующем проекте жизненного цикла.

Продолжение: [FastAPI-сервис с Prometheus и Grafana](https://github.com/matevosovp/ML-model_deployment_in_a_cloud_infrastructure).
