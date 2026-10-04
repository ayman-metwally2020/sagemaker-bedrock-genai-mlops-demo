## Model Training

This folder focuses on model training logic, algorithm selection, and hyperparameter tuning using Amazon SageMaker.

**Algorithm:** SageMaker built-in XGBoost (`reg:squarederror`). It handles non-linear KPI interactions well, trains in minutes, and needs no custom container.

| Hyperparameter | Value | Why |
|---|---|---|
| `num_round` | 300 | With early stopping (20 rounds) on the validation set |
| `max_depth` | 6 | Captures KPI interactions without overfitting |
| `eta` | 0.1 | Standard learning rate |
| `subsample` | 0.8 | Reduces variance |

Next step: wrap the training step in a SageMaker `HyperparameterTuner` to search `max_depth`, `eta` and `min_child_weight` against validation RMSE.
