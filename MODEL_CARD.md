# Model card: real-estate price regression

## Intended use

The model estimates residential property price from structured listing and building attributes. It is designed as an MLOps portfolio example and as an upstream model for the linked inference service.

It is not a certified valuation, lending decision system or substitute for a professional appraisal.

## Data contract

The source is `public.real_estate_dataset_clean` in PostgreSQL. The versioned contract contains 17 input columns and the `price` target. `flat_id` and `building_id` are used for lineage and group splitting but are excluded from the serving signature.

Important feature groups:

- area and room characteristics;
- floor and building characteristics;
- construction year and building type;
- latitude and longitude;
- apartment, studio and elevator flags.

## Validation policy

- `building_id` is the split group, preventing the same building from appearing in multiple partitions;
- approximately 60% of rows are used for training, 20% for validation and 20% for final testing;
- Optuna and Randomized Search only use train and validation;
- the selected pipeline is refit on train+validation;
- test labels are read once for the final candidate evaluation;
- every run logs the dataset fingerprint, random seed, split sizes and group counts.

RMSE, MAE and R² are stored in MLflow instead of being copied into this file. This prevents a stale model card from disagreeing with the registered model version.

## Model and serving contract

The Registry artifact is a complete sklearn Pipeline: deterministic domain features, imputation, unknown-category handling, feature selection and CatBoost regression. Numeric signature fields use a nullable-compatible floating-point contract. The `candidate` alias points to the latest reviewed candidate; promotion to `champion` requires a separate quality decision.

## Limitations and risks

- quality may degrade outside the geography, price range and time period represented in training data;
- market drift can quickly invalidate price estimates;
- location variables can encode socioeconomic patterns and should not be repurposed for eligibility decisions;
- the current workflow reports point estimates without calibrated prediction intervals;
- group splitting measures generalization to unseen buildings but is not a time-based backtest;
- upstream schema or unit changes must be treated as breaking data-contract changes.

## Monitoring recommendations

For deployment, monitor input schema failures, missing-value rates, geographic and price drift, prediction latency and delayed-error metrics when actual sale prices become available. Retraining should create a new Registry version rather than overwriting an existing artifact.
