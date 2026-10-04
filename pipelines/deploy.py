"""Deploy the latest *Approved* model from the Model Registry to a real-time endpoint.

Data capture is switched on so SageMaker Model Monitor can compare live traffic
with the training baseline.

Usage:
    python pipelines/deploy.py --role-arn <arn> --endpoint-name telecom-throughput
"""
import argparse

import boto3
import sagemaker
from pipeline import MODEL_PACKAGE_GROUP
from sagemaker import ModelPackage
from sagemaker.model_monitor import DataCaptureConfig


def latest_approved_package(sm_client) -> str:
    resp = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = resp["ModelPackageSummaryList"]
    if not packages:
        raise SystemExit(f"No approved model in '{MODEL_PACKAGE_GROUP}'. Approve one in SageMaker Studio first.")
    return packages[0]["ModelPackageArn"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--endpoint-name", default="telecom-throughput")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--bucket")
    args = parser.parse_args()

    session = sagemaker.Session()
    bucket = args.bucket or session.default_bucket()
    package_arn = latest_approved_package(boto3.client("sagemaker"))

    model = ModelPackage(role=args.role_arn, model_package_arn=package_arn, sagemaker_session=session)
    model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
        data_capture_config=DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=100,
            destination_s3_uri=f"s3://{bucket}/telecom/data-capture",
        ),
    )
    print(f"Deployed {package_arn} to endpoint '{args.endpoint_name}'")


if __name__ == "__main__":
    main()
