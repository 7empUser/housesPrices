# Houses Prices

Сервис по предсказанию цен на квартиру по признакам.

## Технологический стек

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/) [![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.ai/) [![MLflow](https://img.shields.io/badge/MLflow-Tracing-orange.svg)](https://mlflow.org/) [![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/) [![Make](https://img.shields.io/badge/Make-Automation-grey.svg)](https://www.gnu.org/software/make/)

## Установка

### Требования
- [Chocolatey](https://chocolatey.org/) (Windows)
- make (`choco install make`)
- Перезапустите редактор кода после установки

### Быстрый старт
```bash
git clone <repository_url>
cd houses_prices
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
make all_dev               # Установка зависимостей и настройка
```

## Использование
| Задача | Команда |
|--------|---------|
| Предобработка | make preprocessing |
| Обучение | make train |
| Запуск сервера | make run_server_dev |
| Тест API | make test_dev |
| Сборка Docker | make build |

### Пример запроса

```bash
curl -X POST "http://localhost:8000/predict/" \
    -H "Content-Type: application/json" \
    -d @docs/request.json
```

## Структура проекта

- /artifacts    # Сохраненные артефакты (пайплайны, модели, графики)
- /configs      # Настройки
- /data
  - /raw        # Сырые данные
  - /processed  # Обработанные данные
- /docs         # Документы для работы с сервисом
- /notebooks    # Ноутбуки для анализа
- /src
  - /api        # API сервиса
  - /data       # Преобразование данных
  - /model      # Обучение модели и предсказание
- Makefile
- Dockerfile
- README.md
- requirements_dev.txt
- requirements_prod.txt