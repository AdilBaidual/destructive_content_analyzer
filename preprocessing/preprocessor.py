"""
Модуль предобработки текста.
"""

import os
import re
import pandas as pd
import emoji
from stop_words import get_stop_words

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, PREPARED_DATA_DIR

RUSSIAN_STOPWORDS = set(get_stop_words("ru"))


def preprocess_text(text: str) -> str:
    """
    Очистка текста:
    - приведение к нижнему регистру
    - удаление эмодзи, ссылок, упоминаний, хэштегов
    - удаление спецсимволов
    - удаление стоп-слов
    """
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@[\w_]+", "", text)
    text = re.sub(r"#[\w_]+", "", text)
    text = re.sub(r"[^а-яёa-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [w for w in words if w not in RUSSIAN_STOPWORDS]

    return " ".join(words)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Обработка DataFrame с текстами."""
    if "text" not in df.columns:
        raise ValueError("DataFrame должен содержать колонку 'text'")

    df = df.copy()
    df["text_clean"] = df["text"].apply(preprocess_text)
    df = df[df["text_clean"].str.strip().astype(bool)]

    return df


def process_and_save(filename: str) -> str:
    """Обработка файла и сохранение результата."""
    input_path = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    df = pd.read_csv(input_path)

    if "text_clean" not in df.columns:
        df = process_dataframe(df)

    os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
    name_without_ext = os.path.splitext(filename)[0]
    output_filename = f"{name_without_ext}_prepared.csv"
    output_path = os.path.join(PREPARED_DATA_DIR, output_filename)

    df.to_csv(output_path, index=False)

    return output_filename
