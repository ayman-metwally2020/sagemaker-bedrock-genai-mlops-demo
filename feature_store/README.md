## Feature Store

This folder demonstrates how features are defined and managed using Amazon SageMaker Feature Store to ensure consistency between training and inference.

```bash
python feature_store/create_feature_group.py --role-arn $ROLE --csv data/telecom_kpis.csv
```

Creates the `telecom-cell-kpis` feature group with both stores enabled and ingests the CSV. Query the offline store with Athena for training sets; read the online store with `GetRecord` at inference time.
