## Monitoring and Drift Detection

`setup_monitoring.py` sets up the production feedback loop:

1. **Baseline**: profiles the training data to produce `statistics.json` and `constraints.json`.
2. **Schedule**: runs an hourly data-quality job against data captured from the endpoint.
3. **Metrics**: publishes per-feature drift metrics to CloudWatch (`aws/sagemaker/Endpoints/data-metrics`).
4. **Alarm**: raises a CloudWatch alarm when `sinr_db` drifts past the threshold. Point it at an SNS topic with `--alarm-topic-arn`, or wire it to an EventBridge rule that starts the SageMaker Pipeline to retrain.

```bash
python monitoring/setup_monitoring.py --role-arn $ROLE --endpoint-name telecom-throughput \
    --baseline-csv s3://$BUCKET/telecom/raw/telecom_kpis.csv
```

Extensions worth adding: model-quality monitoring (with ground-truth labels), and Clarify feature-attribution drift.
