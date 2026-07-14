# Security policy

## Reporting

Please report suspected vulnerabilities through GitHub's private security advisory flow. Do not open a public issue containing credentials, bucket names, database addresses or exploit details.

## Secrets and infrastructure

- Keep real values only in the untracked `.env` or in the deployment secret manager.
- Prefer IAM/workload identity over long-lived S3 access keys.
- Keep MLflow bound to loopback unless it is protected by authentication, TLS and a reverse proxy.
- Rotate any credential immediately if it is committed, printed in CI logs or shared in an artifact.

## Model artifacts

Python model formats can execute code during deserialization. `register_baseline.py` therefore requires a trusted SHA-256 digest and verifies the downloaded bytes before calling `joblib.load`. Only approve a digest obtained through a trusted release channel.
