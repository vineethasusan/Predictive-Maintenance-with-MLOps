# Deploying to Azure

The prediction API runs on **Azure Container Apps** (serverless containers). With the
consumption plan and `min-replicas 0` it scales to zero when idle, so a portfolio demo costs
little or nothing within Azure's monthly free grant.

## One-time setup

1. Create a free Azure account at azure.microsoft.com/free.
2. Install the Azure CLI and sign in: `az login`.
3. From the repo root run:

```bash
make deploy          # or: bash deploy/deploy_azure.sh
```

This creates a resource group in **UAE North**, builds the Docker image in Azure Container
Registry and deploys it. The script prints the public URL; open `/docs` to try the API.

## Production notes

| Concern | Approach |
|---|---|
| Model source | Load from the MLflow Model Registry (`models:/fleet-breakdown-risk/Production`) instead of baking into the image |
| Secrets | Azure Key Vault references in Container Apps, never in the image |
| Scheduling | Airflow (or Azure Data Factory / Databricks Workflows) runs `dags/fleet_pm_pipeline.py` weekly |
| Monitoring | Container Apps logs to Log Analytics; `/metrics` endpoint for request counts and latency; weekly PSI drift report |
| Rollback | Container Apps keeps revisions; switch traffic back to the previous revision in one command |

## Clean up

```bash
az group delete --name rg-fleet-pm --yes
```
