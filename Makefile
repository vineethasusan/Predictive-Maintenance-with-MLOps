export PYTHONPATH := src

.PHONY: install data train score monitor api test mlflow-ui up deploy

install:   ## install dependencies
	pip install -r requirements-dev.txt
data:      ## generate 1 year of synthetic telemetry for 400 vehicles
	python -m fleetpm.data
train:     ## train, compare, gate and save the model (logs to MLflow)
	python -m fleetpm.train
score:     ## daily workshop priority list
	python -m fleetpm.score_fleet
monitor:   ## data drift report (PSI)
	python -m fleetpm.monitor
api:       ## serve predictions on :8000 (docs at /docs)
	uvicorn serve.api:app --reload
test:
	pytest -q
mlflow-ui: ## browse experiments on :5000
	mlflow ui
up:        ## API + MLflow + Airflow with Docker
	docker compose up --build
deploy:    ## deploy the API to Azure Container Apps (see deploy/azure.md)
	bash deploy/deploy_azure.sh
