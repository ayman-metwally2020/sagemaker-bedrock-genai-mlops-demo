"""Define and (optionally) run the SageMaker Pipeline.

Steps: Preprocess -> Train (XGBoost) -> Evaluate -> if RMSE <= threshold, register
the model in the Model Registry as "PendingManualApproval".

Usage:
    python pipelines/pipeline.py --role-arn arn:aws:iam::<acct>:role/<SageMakerRole> \
        --bucket <bucket> --upsert --start
"""
import argparse
import os

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_NAME = "telecom-throughput-pipeline"
MODEL_PACKAGE_GROUP = "telecom-throughput-models"


def build_pipeline(role_arn: str, bucket: str, session: sagemaker.Session) -> Pipeline:
    region = session.boto_region_name

    input_data = ParameterString("InputDataUrl", default_value=f"s3://{bucket}/telecom/raw/")
    instance_type = ParameterString("InstanceType", default_value="ml.m5.xlarge")
    rmse_threshold = ParameterFloat("RmseThreshold", default_value=15.0)

    # 1. Preprocess
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=session,
    )
    step_process = ProcessingStep(
        name="Preprocess",
        processor=processor,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name=name, source=f"/opt/ml/processing/{name}")
            for name in ("train", "validation", "test")
        ],
        code=os.path.join(HERE, "preprocess.py"),
    )

    # 2. Train
    xgb_image = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")
    estimator = Estimator(
        image_uri=xgb_image,
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        output_path=f"s3://{bucket}/telecom/models/",
        sagemaker_session=session,
        hyperparameters={
            "objective": "reg:squarederror",
            "num_round": 300,
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "early_stopping_rounds": 20,
        },
    )
    outputs = step_process.properties.ProcessingOutputConfig.Outputs
    step_train = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(outputs["train"].S3Output.S3Uri, content_type="text/csv"),
            "validation": TrainingInput(outputs["validation"].S3Output.S3Uri, content_type="text/csv"),
        },
    )

    # 3. Evaluate
    eval_processor = ScriptProcessor(
        image_uri=xgb_image,
        command=["python3"],
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=session,
    )
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    step_eval = ProcessingStep(
        name="Evaluate",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(source=outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test"),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        code=os.path.join(HERE, "evaluate.py"),
        property_files=[evaluation_report],
    )

    # 4. Register if good enough
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"],
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )
    step_register = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )
    step_condition = ConditionStep(
        name="CheckRmse",
        conditions=[
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=step_eval.name,
                    property_file=evaluation_report,
                    json_path="regression_metrics.rmse.value",
                ),
                right=rmse_threshold,
            )
        ],
        if_steps=[step_register],
        else_steps=[],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data, instance_type, rmse_threshold],
        steps=[step_process, step_train, step_eval, step_condition],
        sagemaker_session=session,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--bucket", help="Defaults to the SageMaker default bucket")
    parser.add_argument("--upsert", action="store_true", help="Create or update the pipeline in SageMaker")
    parser.add_argument("--start", action="store_true", help="Start an execution after upserting")
    args = parser.parse_args()

    session = sagemaker.Session()
    bucket = args.bucket or session.default_bucket()
    pipeline = build_pipeline(args.role_arn, bucket, session)

    if not args.upsert:
        print(pipeline.definition())
        return
    pipeline.upsert(role_arn=args.role_arn)
    print(f"Pipeline '{PIPELINE_NAME}' upserted")
    if args.start:
        execution = pipeline.start()
        print(f"Started execution: {execution.arn}")


if __name__ == "__main__":
    main()
