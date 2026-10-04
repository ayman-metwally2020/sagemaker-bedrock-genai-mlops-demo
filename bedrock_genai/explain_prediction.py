"""Turn a model prediction into a plain-language explanation for NOC engineers,
using Claude on Amazon Bedrock.

Usage (with a live endpoint):
    python bedrock_genai/explain_prediction.py --endpoint-name telecom-throughput \
        --kpis '{"prb_utilization": 92, "active_users": 180, "rsrp_dbm": -112, "sinr_db": 2.5, "handover_failure_rate": 0.06, "hour_of_day": 20}'

Usage (without an endpoint, pass the prediction yourself):
    python bedrock_genai/explain_prediction.py --prediction 38.4 --kpis '{...}'
"""
import argparse
import json
import os

import boto3
from anthropic import AnthropicBedrockMantle

FEATURE_ORDER = [
    "prb_utilization",
    "active_users",
    "rsrp_dbm",
    "sinr_db",
    "handover_failure_rate",
    "hour_of_day",
]
MODEL_ID = "anthropic.claude-opus-5-5"

SYSTEM_PROMPT = """You are an assistant for a mobile network operations centre (NOC).
You receive one cell's KPIs and an ML model's predicted downlink throughput.
Explain to an on-shift engineer, in under 150 words:
1. Whether the predicted throughput is healthy, degraded or critical (healthy > 80 Mbps, critical < 30 Mbps).
2. The two or three KPIs most likely driving the prediction, with their values.
3. One or two concrete next checks or actions.
Use plain language and short bullet points. Do not invent KPIs that were not provided."""


def predict(endpoint_name: str, kpis: dict) -> float:
    row = ",".join(str(kpis[f]) for f in FEATURE_ORDER)
    runtime = boto3.client("sagemaker-runtime")
    resp = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType="text/csv", Body=row)
    return float(resp["Body"].read().decode().strip().split(",")[0])


def build_prompt(kpis: dict, prediction: float) -> str:
    return (
        "Cell KPIs:\n"
        + json.dumps(kpis, indent=2)
        + f"\n\nPredicted downlink throughput: {prediction:.1f} Mbps"
    )


def explain(kpis: dict, prediction: float, region: str) -> str:
    client = AnthropicBedrockMantle(aws_region=region)
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=2000,
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(kpis, prediction)}],
    )
    if response.stop_reason == "refusal":
        return "The model declined to produce an explanation for this input."
    return "".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kpis", required=True, help="JSON object with the six KPI fields")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--endpoint-name")
    group.add_argument("--prediction", type=float)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    args = parser.parse_args()

    kpis = json.loads(args.kpis)
    missing = [f for f in FEATURE_ORDER if f not in kpis]
    if missing:
        raise SystemExit(f"Missing KPI fields: {missing}")

    prediction = args.prediction if args.prediction is not None else predict(args.endpoint_name, kpis)
    print(f"Predicted throughput: {prediction:.1f} Mbps\n")
    print(explain(kpis, prediction, args.region))


if __name__ == "__main__":
    main()
