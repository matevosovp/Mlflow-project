#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Автоподгрузка .env из корня репозитория (если есть)
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
: "${DB_DESTINATION_DBNAME:?DB_DESTINATION_DBNAME is required}"

: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID is required}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY is required}"
: "${MLFLOW_S3_ENDPOINT_URL:?MLFLOW_S3_ENDPOINT_URL is required}"

MLFLOW_SERVER_HOST="${MLFLOW_SERVER_HOST:-0.0.0.0}"
MLFLOW_SERVER_PORT="${MLFLOW_SERVER_PORT:-5000}"

# Важно: если пароль содержит спецсимволы, задайте DB_DESTINATION_PASSWORD уже urlencoded
BACKEND_URI="postgresql+psycopg2://${DB_DESTINATION_USER}:${DB_DESTINATION_PASSWORD}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_DBNAME}"


BUCKET_NAME="${S3_BUCKET_NAME}"
ARTIFACT_ROOT="s3://${BUCKET_NAME}/mlflow"

echo "Starting MLflow Server"
echo "  host: ${MLFLOW_SERVER_HOST}"
echo "  port: ${MLFLOW_SERVER_PORT}"
echo "  backend-store-uri: ${BACKEND_URI}"
echo "  registry-store-uri: ${BACKEND_URI}"
echo "  default-artifact-root: ${ARTIFACT_ROOT}"
echo "  s3 endpoint: ${MLFLOW_S3_ENDPOINT_URL}"

exec mlflow server \
  --host "${MLFLOW_SERVER_HOST}" \
  --port "${MLFLOW_SERVER_PORT}" \
  --backend-store-uri "${BACKEND_URI}" \
  --registry-store-uri "${BACKEND_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --no-serve-artifacts
