## Этап 5. Итог

- experiment: model_improvement_matevosov
- table_name: public.real_estate_dataset_clean
- target: price
- random_state: 42
- final_features_cnt: 29

### Выбор метода на val
- best_method: optuna

### Метрики на test
- rmse_test: 2277225.029537
- mae_test: 1662039.855848
- r2_test: 0.865752
- fit_time_sec: 227.623118
- predict_time_sec: 0.385855

### Лучшие параметры
- best_params_final: {"iterations": 1991, "depth": 10, "learning_rate": 0.1848232926437564, "l2_leaf_reg": 2.1704575003580895, "subsample": 0.6514377826093692, "colsample_bylevel": 0.8176754508303175, "random_strength": 1.2882871141348935, "min_data_in_leaf": 8}
