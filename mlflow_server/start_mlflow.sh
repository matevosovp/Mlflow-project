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
: "${DB_DESTINATION_NAME:?DB_DESTINATION_NAME is required}"

: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID is required}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY is required}"
: "${MLFLOW_S3_ENDPOINT_URL:?MLFLOW_S3_ENDPOINT_URL is required}"


# Важно: если пароль содержит спецсимволы, задайте DB_DESTINATION_PASSWORD уже urlencoded
BACKEND_URI="postgresql+psycopg2://${DB_DESTINATION_USER}:${DB_DESTINATION_PASSWORD}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}"


BUCKET_NAME="${S3_BUCKET_NAME}"
ARTIFACT_ROOT="s3://${BUCKET_NAME}/mlflow"

echo "Starting MLflow Server"
echo "  backend-store-uri: ${BACKEND_URI}"
echo "  registry-store-uri: ${BACKEND_URI}"
echo "  default-artifact-root: ${ARTIFACT_ROOT}"
echo "  s3 endpoint: ${MLFLOW_S3_ENDPOINT_URL}"

exec mlflow server \
  --backend-store-uri "${BACKEND_URI}" \
  --registry-store-uri "${BACKEND_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --no-serve-artifacts


