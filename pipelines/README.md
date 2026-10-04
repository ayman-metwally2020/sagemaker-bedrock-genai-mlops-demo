## SageMaker Pipelines

This folder contains the definition of the automated ML pipeline, including data preparation, training, evaluation, and approval steps.

| File | Purpose |
|---|---|
| `pipeline.py` | Builds the pipeline: Preprocess → Train → Evaluate → register if RMSE ≤ `RmseThreshold` |
| `preprocess.py` | Processing script: cleans data and writes train/validation/test splits |
| `evaluate.py` | Processing script: computes RMSE, MAE and R² into `evaluation.json` |
| `deploy.py` | Deploys the latest **Approved** model package with data capture enabled |

```bash
# Print the pipeline JSON definition without creating anything
python pipelines/pipeline.py --role-arn $ROLE --bucket $BUCKET

# Create/update and start it
python pipelines/pipeline.py --role-arn $ROLE --bucket $BUCKET --upsert --start
```
