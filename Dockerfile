# Dev

FROM python:3.13-slim AS dev

WORKDIR /app

COPY requirements_dev.txt .
RUN pip install --no-cache-dir --only-binary :all: \
    --timeout 60 --retries 5 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements_dev.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]



# Prod

FROM python:3.13-slim AS prod

WORKDIR /app

COPY requirements_prod.txt .
RUN pip install --no-cache-dir --only-binary :all: \
    --timeout 60 --retries 5 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements_prod.txt

COPY artifacts/models/model.json /app/artifacts/models/
COPY artifacts/preprocessors/preprocessor.joblib /app/artifacts/preprocessors/
COPY configs/__init__.py /app/configs/
COPY configs/paths.py /app/configs/
COPY src/__init__.py /app/src/__init__.py
COPY src/api /app/src/api
COPY src/model/predict.py /app/src/model/predict.py

EXPOSE 80

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "80"]