## Этап 5. Итог

- experiment: model_FE_matevosov
- table_name: public.real_estate_dataset_clean
- target: price
- random_state: 42
- final_features_cnt: 33

### Выбор метода на val
- best_method: search

### Метрики на test
- rmse_test: 2307440.754192
- mae_test: 1732828.400482
- r2_test: 0.862165
- fit_time_sec: 133.538406
- predict_time_sec: 0.167854

### Лучшие параметры
- best_params_final: {"subsample": 0.7, "min_data_in_leaf": 64, "learning_rate": 0.1, "l2_leaf_reg": 1, "iterations": 800, "depth": 10, "colsample_bylevel": 1.0}
