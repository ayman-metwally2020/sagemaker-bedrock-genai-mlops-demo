"""Generate a synthetic telecom cell-KPI dataset.

Each row is one cell at one hour. The target, ``throughput_mbps``, depends on
radio quality, load and failures, so a regression model has real signal to learn.

Usage:
    python data/generate_data.py --rows 20000 --out data/telecom_kpis.csv
"""
import argparse

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


def generate(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_cells = max(1, rows // 24)

    cell_id = rng.integers(0, n_cells, rows)
    hour = rng.integers(0, 24, rows)
    # Busy-hour load curve peaking around 20:00.
    busy = 0.5 + 0.5 * np.sin((hour - 14) / 24 * 2 * np.pi)

    prb = np.clip(rng.normal(35 + 45 * busy, 10), 1, 100)
    users = np.clip(rng.poisson(20 + 120 * busy), 1, None)
    rsrp = np.clip(rng.normal(-95, 10, rows), -130, -60)
    sinr = np.clip(rng.normal(12, 6, rows), -5, 30)
    ho_fail = np.clip(rng.beta(1.2, 40, rows), 0, 1)

    throughput = (
        150
        + 4.0 * sinr
        + 0.8 * (rsrp + 95)
        - 0.9 * prb
        - 0.25 * users
        - 250 * ho_fail
        + rng.normal(0, 8, rows)
    )
    throughput = np.clip(throughput, 1, None)

    return pd.DataFrame(
        {
            "cell_id": [f"CELL-{c:04d}" for c in cell_id],
            "hour_of_day": hour,
            "prb_utilization": prb.round(2),
            "active_users": users,
            "rsrp_dbm": rsrp.round(1),
            "sinr_db": sinr.round(1),
            "handover_failure_rate": ho_fail.round(4),
            TARGET: throughput.round(2),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/telecom_kpis.csv")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
