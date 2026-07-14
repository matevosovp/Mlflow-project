#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

: "${DB_DESTINATION_HOST:?DB_DESTINATION_HOST is required}"
: "${DB_DESTINATION_PORT:?DB_DESTINATION_PORT is required}"
: "${DB_DESTINATION_USER:?DB_DESTINATION_USER is required}"
: "${DB_DESTINATION_PASSWORD:?DB_DESTINATION_PASSWORD is required}"
: "${DB_DESTINATION_NAME:?DB_DESTINATION_NAME is required}"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID is required}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY is required}"
: "${MLFLOW_S3_ENDPOINT_URL:?MLFLOW_S3_ENDPOINT_URL is required}"
: "${S3_BUCKET_NAME:?S3_BUCKET_NAME is required}"

MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"

BACKEND_URI="$(
python - <<'PY'
import os
from urllib.parse import quote_plus

user = quote_plus(os.environ["DB_DESTINATION_USER"])
password = quote_plus(os.environ["DB_DESTINATION_PASSWORD"])
host = os.environ["DB_DESTINATION_HOST"]
port = os.environ["DB_DESTINATION_PORT"]
database = os.environ["DB_DESTINATION_NAME"]

print(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")
PY
)"

ARTIFACT_ROOT="s3://${S3_BUCKET_NAME}/mlflow"

echo "Starting MLflow Server"
echo "  backend: PostgreSQL ${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}"
echo "  artifact root: s3://${S3_BUCKET_NAME}/mlflow"
echo "  listen: ${MLFLOW_HOST}:${MLFLOW_PORT}"

exec mlflow server \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --backend-store-uri "${BACKEND_URI}" \
  --registry-store-uri "${BACKEND_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --no-serve-artifacts
