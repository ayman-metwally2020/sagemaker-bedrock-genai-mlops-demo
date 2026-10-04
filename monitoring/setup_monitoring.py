"""Baseline the training data, schedule hourly data-quality monitoring, and add a
CloudWatch alarm on feature drift.

Model Monitor publishes per-feature drift metrics to CloudWatch. The alarm can
feed an EventBridge rule that starts the SageMaker Pipeline to retrain.

Usage:
    python monitoring/setup_monitoring.py --role-arn <arn> --endpoint-name telecom-throughput \
        --baseline-csv s3://<bucket>/telecom/baseline/train_with_header.csv
"""
import argparse

import boto3
import sagemaker
from sagemaker.model_monitor import CronExpressionGenerator, DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

# Feature watched by the alarm; SINR shifts are an early sign of radio degradation.
ALARM_FEATURE = "sinr_db"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--endpoint-name", default="telecom-throughput")
    parser.add_argument("--baseline-csv", required=True, help="Training data CSV with a header row, in S3")
    parser.add_argument("--bucket")
    parser.add_argument("--alarm-topic-arn", help="Optional SNS topic for the drift alarm")
    args = parser.parse_args()

    session = sagemaker.Session()
    bucket = args.bucket or session.default_bucket()
    prefix = f"s3://{bucket}/telecom/monitoring"

    monitor = DefaultModelMonitor(
        role=args.role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=session,
    )

    print("Running baseline job (statistics + constraints)...")
    monitor.suggest_baseline(
        baseline_dataset=args.baseline_csv,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=f"{prefix}/baseline",
        wait=True,
    )

    schedule_name = f"{args.endpoint_name}-data-quality"
    monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule_name,
        endpoint_input=args.endpoint_name,
        output_s3_uri=f"{prefix}/reports",
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    )
    print(f"Monitoring schedule '{schedule_name}' created")

    boto3.client("cloudwatch").put_metric_alarm(
        AlarmName=f"{args.endpoint_name}-{ALARM_FEATURE}-drift",
        Namespace="aws/sagemaker/Endpoints/data-metrics",
        MetricName=f"feature_baseline_drift_{ALARM_FEATURE}",
        Dimensions=[
            {"Name": "Endpoint", "Value": args.endpoint_name},
            {"Name": "MonitoringSchedule", "Value": schedule_name},
        ],
        Statistic="Average",
        Period=3600,
        EvaluationPeriods=1,
        Threshold=0.2,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[args.alarm_topic_arn] if args.alarm_topic_arn else [],
        AlarmDescription="Live SINR distribution drifted from the training baseline; consider retraining.",
    )
    print("CloudWatch drift alarm created")


if __name__ == "__main__":
    main()
