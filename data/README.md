## Data

This folder represents sample or anonymized datasets used for model training and testing. In real deployments, data would be sourced from secure storage such as Amazon S3 and processed according to governance and compliance requirements.

```bash
python data/generate_data.py --rows 20000 --out data/telecom_kpis.csv
```

The generator produces realistic, fully synthetic hourly cell KPIs: busy-hour load curves, RSRP/SINR distributions, and handover failures. Columns: `cell_id`, `hour_of_day`, `prb_utilization`, `active_users`, `rsrp_dbm`, `sinr_db`, `handover_failure_rate`, `throughput_mbps` (target).
