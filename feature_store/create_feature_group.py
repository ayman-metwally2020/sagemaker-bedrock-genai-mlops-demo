"""Create a SageMaker Feature Store feature group for cell KPIs and ingest data.

The same feature group serves training (offline store, queried with Athena) and
real-time inference (online store), so both paths use identical feature values.

Usage:
    python feature_store/create_feature_group.py --role-arn <arn> --csv data/telecom_kpis.csv
"""
import argparse
import time

import pandas as pd
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

FEATURE_GROUP_NAME = "telecom-cell-kpis"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Feature Store needs a unique record id and an event time per record.
    df["record_id"] = df["cell_id"] + "-" + df["hour_of_day"].astype(str) + "-" + df.index.astype(str)
    df["event_time"] = float(round(time.time()))
    df["cell_id"] = df["cell_id"].astype("string")
    df["record_id"] = df["record_id"].astype("string")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--csv", default="data/telecom_kpis.csv")
    parser.add_argument("--bucket", help="Offline store bucket (defaults to SageMaker default bucket)")
    args = parser.parse_args()

    session = sagemaker.Session()
    bucket = args.bucket or session.default_bucket()
    df = prepare(pd.read_csv(args.csv))

    fg = FeatureGroup(name=FEATURE_GROUP_NAME, sagemaker_session=session)
    fg.load_feature_definitions(data_frame=df)
    fg.create(
        s3_uri=f"s3://{bucket}/telecom/feature-store",
        record_identifier_name="record_id",
        event_time_feature_name="event_time",
        role_arn=args.role_arn,
        enable_online_store=True,
    )
    while fg.describe()["FeatureGroupStatus"] == "Creating":
        print("Waiting for feature group...")
        time.sleep(10)

    fg.ingest(data_frame=df, max_workers=4, wait=True)
    print(f"Ingested {len(df)} records into {FEATURE_GROUP_NAME}")


if __name__ == "__main__":
    main()
