"""
Конфигурация проекта.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Telegram API
TG_API_ID = int(os.getenv("TG_API_ID", "0"))
TG_API_HASH = os.getenv("TG_API_HASH", "")

# Директории
DATA_DIR = "data"
RAW_DATA_DIR = "raw_data"
PREPARED_DATA_DIR = "prepared_data"
STATIC_DIR = "static"

# Датасеты
TOXIC_COMMENTS_PATH = os.path.join(DATA_DIR, "toxic_comments.csv")
EXTREMISM_DATA_PATH = os.path.join(DATA_DIR, "extremism_data.csv")
ADDITIONAL_DESTRUCTIVE_PATH = os.path.join(DATA_DIR, "additional_destructive.csv")
ADDITIONAL_EXTREMISM_PATH = os.path.join(DATA_DIR, "additional_extremism.csv")

# Модель деструктива (sklearn)
DESTRUCTIVE_MODEL_DIR = "models/destructive"
DESTRUCTIVE_MODEL_PATH = os.path.join(DESTRUCTIVE_MODEL_DIR, "model.pkl")
DESTRUCTIVE_VECTORIZER_PATH = os.path.join(DESTRUCTIVE_MODEL_DIR, "vectorizer.pkl")

# Модель экстремизма (Keras)
EXTREMISM_MODEL_DIR = "models/extremism"
EXTREMISM_MODEL_PATH = os.path.join(EXTREMISM_MODEL_DIR, "extremism_model.keras")
EXTREMISM_TOKENIZER_PATH = os.path.join(EXTREMISM_MODEL_DIR, "tokenizer.json")
EXTREMISM_MAXLEN_PATH = os.path.join(EXTREMISM_MODEL_DIR, "maxlen.txt")

# Параметры моделей
TFIDF_MAX_FEATURES = 5000
KERAS_VOCAB_SIZE = 1000
KERAS_EMBEDDING_DIM = 16
KERAS_MAXLEN_DEFAULT = 50
