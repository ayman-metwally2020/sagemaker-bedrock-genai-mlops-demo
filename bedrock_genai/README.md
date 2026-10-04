## GenAI with AWS Bedrock

This folder shows how AWS Bedrock can be used to generate human-readable explanations from model predictions, helping engineers and decision-makers understand and act on ML outputs.

`explain_prediction.py` takes a cell's KPIs and either calls the SageMaker endpoint or accepts a prediction you pass in. It then asks Claude on Amazon Bedrock for a short NOC-ready summary: status, likely drivers, and next checks.

```bash
# No endpoint needed: pass a prediction directly
python bedrock_genai/explain_prediction.py --prediction 38.4 \
    --kpis '{"prb_utilization": 92, "active_users": 180, "rsrp_dbm": -112, "sinr_db": 2.5, "handover_failure_rate": 0.06, "hour_of_day": 20}'
```

Requirements: Bedrock model access for Claude enabled in your AWS region, and AWS credentials with `bedrock:InvokeModel`.
