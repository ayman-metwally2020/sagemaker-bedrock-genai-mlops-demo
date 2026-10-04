## Architecture

```mermaid
flowchart TB
    subgraph Data
        S3[(S3: raw KPIs)]
        FS[(Feature Store<br/>online + offline)]
    end
    subgraph Build["SageMaker Pipeline"]
        P1[Processing: preprocess.py] --> P2[Training: XGBoost]
        P2 --> P3[Processing: evaluate.py]
        P3 --> P4{RMSE <= threshold}
        P4 -- yes --> REG[Model Registry<br/>PendingManualApproval]
    end
    subgraph Serve
        EP[Real-time endpoint<br/>data capture on]
        MM[Model Monitor<br/>hourly data quality]
        CW[CloudWatch alarm]
        EB[EventBridge rule]
    end
    subgraph Explain
        BR[Claude on Amazon Bedrock]
        NOC[NOC engineer]
    end

    S3 --> P1
    S3 --> FS
    REG -- approved --> EP
    EP --> MM --> CW --> EB -. start pipeline .-> P1
    EP -- prediction + KPIs --> BR --> NOC
```

### Flow

1. **Ingest**: raw hourly cell KPIs land in S3 and are optionally written to Feature Store, so training and inference use the same feature definitions.
2. **Build**: the pipeline cleans and splits data, trains XGBoost, evaluates on a held-out test set, and registers the model only if RMSE is acceptable.
3. **Approve and deploy**: a human approves the model package; `pipelines/deploy.py` deploys the latest approved version with data capture enabled.
4. **Monitor**: Model Monitor compares captured traffic against the training baseline every hour and publishes drift metrics to CloudWatch.
5. **Retrain**: a drift alarm can drive an EventBridge rule that restarts the pipeline.
6. **Explain**: each prediction plus its KPIs is sent to Claude on Bedrock, which returns a short status, likely drivers, and next actions.

### Security notes

- Use a least-privilege SageMaker execution role scoped to the project bucket.
- Keep endpoints in a VPC with private subnets for production; use VPC endpoints for S3, SageMaker and Bedrock.
- Encrypt S3, Feature Store and endpoint volumes with KMS.
