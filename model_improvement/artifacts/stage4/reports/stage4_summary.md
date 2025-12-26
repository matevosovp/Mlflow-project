## Этап 4. Итог

- experiment: model_improvement_matevosov
- run_name: 04_feature_selection
- table_name: public.real_estate_dataset_clean
- target: price
- random_state: 42

### Выбор набора признаков по val
- best_set: Union
- n_features: 29
- rmse_val(best): 2265245.5953

### Финальная оценка на test
- rmse_test: 2418985.8215
- mae_test: 1841964.7591
- r2_test: 0.8485
- fit_time_sec: 20.1053
- predict_time_sec: 0.0209
