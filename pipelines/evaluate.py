"""SageMaker Processing step: score the trained XGBoost model on the test split.

Writes ``evaluation.json`` in the SageMaker Model Registry metrics format; the
pipeline's condition step reads ``regression_metrics.rmse.value`` from it.
"""
import json
import pathlib
import tarfile

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {
        "regression_metrics": {
            "rmse": {"value": rmse},
            "mae": {"value": mae},
            "r2": {"value": r2},
        }
    }


def main() -> None:
    import xgboost  # available in the SageMaker XGBoost image

    with tarfile.open("/opt/ml/processing/model/model.tar.gz") as tar:
        tar.extractall(path=".")
    booster = xgboost.Booster()
    booster.load_model("xgboost-model")

    test = pd.read_csv("/opt/ml/processing/test/test.csv", header=None)
    y_true = test.iloc[:, 0].to_numpy()
    preds = booster.predict(xgboost.DMatrix(test.iloc[:, 1:].values))

    report = regression_metrics(y_true, preds)
    out_dir = pathlib.Path("/opt/ml/processing/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
