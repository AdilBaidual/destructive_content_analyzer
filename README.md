# Анализатор деструктивного контента

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_Framework-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-DL-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org/)

Веб-приложение для анализа Telegram-каналов на деструктивный контент. Умеет парсить посты, строить графики, определять экстремизм и дообучаться на новых данных.

## Что умеет

- Анализировать посты из Telegram-каналов
- Строить графики динамики и распределений
- Выделять самые деструктивные посты
- Отдельный отчёт по экстремизму
- Дообучать модели

## Структура

```
destructive_content_analyzer/
├── app.py                   # Flask приложение
├── config.py                # Конфигурация
├── requirements.txt         # Зависимости
│
├── parser/                  # Парсинг Telegram
│   └── telegram_parser.py
│
├── preprocessing/           # Обработка текста
│   └── text_preprocessor.py
│
├── models/                  # ML модели
│   ├── trainer.py
│   ├── destructive/
│   └── extremism/
│
├── analysis/                # Анализ и графики
│   ├── analyzer.py
│   └── visualizer.py
│
├── data/                    # Датасеты
│   ├── raw/
│   └── processed/
│
├── static/                  # CSS/JS/графики
└── templates/               # HTML шаблоны
```

## Как устроены модели

| Модель | Алгоритм | Задача |
|--------|----------|--------|
| Деструктивность | SGDClassifier + TF-IDF | Бинарная классификация |
| Экстремизм | Keras Sequential | Нейросеть |

**Препроцессинг:**
1. Удаляем ссылки и упоминания
2. Лемматизация через pymorphy2
3. TF-IDF векторизация

## Запуск

```bash
# Создать окружение
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Настроить Telegram API
cp .env.example .env
# Получить API_ID и API_HASH на https://my.telegram.org

# Обучить модели
python models/trainer.py

# Запустить
python app.py
```

Будет доступно на `http://localhost:5000`

## Метрики

**Деструктивность:**
- Precision: 0.87, Recall: 0.82, F1: 0.84

**Экстремизм:**
- Accuracy: 0.91, F1: 0.88

## API

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/` | Главная |
| POST | `/analyze` | Анализ канала |
| GET | `/results/<id>` | Результаты |
| POST | `/train` | Дообучение |

## Зависимости

```
flask>=2.3.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
telethon>=1.29.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
wordcloud>=1.9.0
pymorphy2>=0.9.0
plotly>=5.15.0
```

## Требования

- Python 3.11+
- 4GB RAM (для TensorFlow)
- Telegram API credentials
