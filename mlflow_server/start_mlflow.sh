#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${ENV_FILE}" ]]; then
  # Parse dotenv values as data. shlex.quote makes the generated export statements
  # safe even when passwords contain spaces or shell metacharacters.
  eval "$("${PYTHON_BIN}" - "${ENV_FILE}" <<'PY'
import re
import shlex
import sys
from dotenv import dotenv_values

for key, value in dotenv_values(sys.argv[1]).items():
    if value is None:
        continue
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise SystemExit(f"Invalid environment variable name in .env: {key!r}")
    print(f"export {key}={shlex.quote(value)}")
PY
)"
fi

required_vars=(
  DB_DESTINATION_HOST
  DB_DESTINATION_PORT
  DB_DESTINATION_NAME
  DB_DESTINATION_USER
  DB_DESTINATION_PASSWORD
  MLFLOW_S3_ENDPOINT_URL
  S3_BUCKET_NAME
)

for variable in "${required_vars[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Required environment variable is missing: ${variable}" >&2
    exit 1
  fi
done

MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"

BACKEND_URI="$("${PYTHON_BIN}" - <<'PY'
import os
from sqlalchemy import URL

url = URL.create(
    "postgresql+psycopg",
    username=os.environ["DB_DESTINATION_USER"],
    password=os.environ["DB_DESTINATION_PASSWORD"],
    host=os.environ["DB_DESTINATION_HOST"],
    port=int(os.environ["DB_DESTINATION_PORT"]),
    database=os.environ["DB_DESTINATION_NAME"],
)
print(url.render_as_string(hide_password=False))
PY
)"
ARTIFACT_ROOT="s3://${S3_BUCKET_NAME}"

echo "Starting MLflow on ${MLFLOW_HOST}:${MLFLOW_PORT}"
echo "Backend: PostgreSQL at ${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}"
echo "Artifacts: ${ARTIFACT_ROOT} via ${MLFLOW_S3_ENDPOINT_URL}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m mlflow server \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --backend-store-uri "${BACKEND_URI}" \
  --registry-store-uri "${BACKEND_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --no-serve-artifacts
