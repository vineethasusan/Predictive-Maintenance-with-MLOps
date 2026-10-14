#!/usr/bin/env bash
# Deploy the prediction API to Azure Container Apps (consumption plan, scales to zero).
# Prerequisites: Azure CLI installed and `az login` done.
set -euo pipefail

RG=${RG:-rg-fleet-pm}
LOCATION=${LOCATION:-uaenorth}
APP=${APP:-fleet-pm-api}
ENVNAME=${ENVNAME:-fleet-pm-env}

az group create --name "$RG" --location "$LOCATION" -o none
az extension add --name containerapp --upgrade -y -o none

# Builds the Dockerfile in Azure (no local Docker needed) and deploys it
az containerapp up \
  --name "$APP" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --environment "$ENVNAME" \
  --source . \
  --ingress external \
  --target-port 8000

az containerapp update --name "$APP" --resource-group "$RG" --min-replicas 0 --max-replicas 2 -o none
URL=$(az containerapp show --name "$APP" --resource-group "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo "Deployed: https://$URL/docs"
