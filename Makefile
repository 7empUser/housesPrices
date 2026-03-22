PIP = pip
PYTHON = python

REQUIREMENTS_DEV = requirements_dev.txt
REQUIREMENTS_PROD = requirements_prod.txt
PREPROCESSING = src.data.preprocessing
TRAIN = src.model.train

install_dev:
	$(PIP) install -r $(REQUIREMENTS_DEV)

preprocessing:
	$(PYTHON) -m $(PREPROCESSING)

train:
	$(PYTHON) -m $(TRAIN)

run_server_dev:
	uvicorn src.api.app:app --host 127.0.0.1 --port 8000

all_dev:
	make install_dev preprocessing train run_server_dev

test_dev:
	curl -X POST "http://localhost:8000/predict/" -H "Content-Type: application/json" -d @docs/request.json

install_prod:
	$(PIP) install -r $(REQUIREMENTS_PROD)

run_server_prod:
	uvicorn src.api.app:app --host 127.0.0.1 --port 80

all_prod:
	make install_prod run_server_prod

test_prod:
	curl -X POST "http://localhost:80/predict/" -H "Content-Type: application/json" -d @docs/request.json

build:
	docker build -t houses_prices .