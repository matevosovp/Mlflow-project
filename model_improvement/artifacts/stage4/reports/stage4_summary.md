## Этап 4. Итог

- experiment: model_FE_matevosov
- run_name: 04_feature_selection
- table_name: public.real_estate_dataset_clean
- target: price
- random_state: 42

### Выбор набора признаков по val
- best_set: Union
- n_features: 33
- rmse_val(best): 2263644.0595

### Финальная оценка на test
- rmse_test: 2413951.5768
- mae_test: 1839931.0593
- r2_test: 0.8491
- fit_time_sec: 20.8227
- predict_time_sec: 0.0221
