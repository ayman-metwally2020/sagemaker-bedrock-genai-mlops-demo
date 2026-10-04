"""SageMaker Processing step: clean the raw KPIs and split train/validation/test.

Output CSVs have the target in the first column and no header, which is the
format the built-in SageMaker XGBoost container expects.
"""
import argparse
import os

import numpy as np
import pandas as pd

FEATURES = [
    "prb_utilization",
    "active_users",
    "rsrp_dbm",
    "sinr_db",
    "handover_failure_rate",
    "hour_of_day",
]
TARGET = "throughput_mbps"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[(df["prb_utilization"].between(0, 100)) & (df[TARGET] > 0)]
    return df[[TARGET] + FEATURES].reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = 42):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end, val_end = int(0.7 * n), int(0.85 * n)
    return np.split(shuffled, [train_end, val_end])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/opt/ml/processing")
    args = parser.parse_args()

    input_dir = os.path.join(args.base_dir, "input")
    frames = [pd.read_csv(os.path.join(input_dir, f)) for f in os.listdir(input_dir) if f.endswith(".csv")]
    df = clean(pd.concat(frames, ignore_index=True))

    for name, part in zip(["train", "validation", "test"], split(df)):
        out_dir = os.path.join(args.base_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        part.to_csv(os.path.join(out_dir, f"{name}.csv"), header=False, index=False)
        print(f"{name}: {len(part)} rows")


if __name__ == "__main__":
    main()
