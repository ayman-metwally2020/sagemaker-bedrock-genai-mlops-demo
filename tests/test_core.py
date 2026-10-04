"""Fast, offline tests: no AWS credentials needed."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("data", "pipelines", "bedrock_genai"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from evaluate import regression_metrics
from generate_data import FEATURES, TARGET, generate
from preprocess import clean, split


def test_generate_shape_and_ranges():
    df = generate(1000, seed=1)
    assert len(df) == 1000
    assert set(FEATURES + [TARGET, "cell_id"]) <= set(df.columns)
    assert df["prb_utilization"].between(0, 100).all()
    assert (df[TARGET] > 0).all()


def test_generate_is_deterministic():
    assert generate(200, seed=7).equals(generate(200, seed=7))


def test_clean_puts_target_first_and_drops_bad_rows():
    df = generate(500)
    df.loc[0, "sinr_db"] = np.nan
    df.loc[1, "prb_utilization"] = 150
    cleaned = clean(df)
    assert list(cleaned.columns) == [TARGET] + FEATURES
    assert len(cleaned) == 498


def test_split_proportions_and_no_overlap():
    cleaned = clean(generate(1000))
    train, val, test = split(cleaned)
    assert (len(train), len(val), len(test)) == (700, 150, 150)
    assert len(train) + len(val) + len(test) == len(cleaned)


def test_regression_metrics_perfect_and_imperfect():
    y = np.array([10.0, 20.0, 30.0])
    perfect = regression_metrics(y, y)["regression_metrics"]
    assert perfect["rmse"]["value"] == 0.0
    assert perfect["r2"]["value"] == 1.0

    off = regression_metrics(y, y + 3)["regression_metrics"]
    assert off["rmse"]["value"] == 3.0
    assert off["mae"]["value"] == 3.0


def test_bedrock_prompt_contains_kpis_and_prediction():
    from explain_prediction import FEATURE_ORDER, build_prompt

    kpis = {f: 1 for f in FEATURE_ORDER}
    prompt = build_prompt(kpis, 42.25)
    assert "42.2 Mbps" in prompt or "42.3 Mbps" in prompt
    for f in FEATURE_ORDER:
        assert f in prompt
