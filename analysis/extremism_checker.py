"""
Модуль проверки на экстремизм.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PREPARED_DATA_DIR, STATIC_DIR
from models.trainer import predict_extremism
from analysis.visualizer import (
    generate_class_distribution,
    generate_probability_histogram,
    generate_wordcloud,
    generate_frequency_chart
)


def check_extremism_file(filename: str, top_n: int = 10) -> Dict[str, Any]:
    """Проверка файла на экстремизм."""
    filepath = os.path.join(PREPARED_DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    df = pd.read_csv(filepath)

    if "text_clean" not in df.columns and "text" not in df.columns:
        raise ValueError("Файл не содержит текстовых данных")

    texts = df["text"].tolist() if "text" in df.columns else df["text_clean"].tolist()
    predictions, probabilities = predict_extremism(texts)

    df["extremism_pred"] = predictions
    df["extremism_prob"] = probabilities

    total = len(df)
    extremist = int((predictions == 1).sum())
    neutral = total - extremist

    top_posts = get_top_extremist_posts(df, top_n=top_n)
    viz_paths = generate_extremism_visualizations(df, predictions, probabilities)

    return {
        "total": total,
        "neutral": neutral,
        "extremist": extremist,
        "extremist_percent": round(extremist / total * 100, 2) if total > 0 else 0,
        "avg_prob": round(float(probabilities.mean()), 4),
        "top_posts": top_posts,
        "visualizations": viz_paths,
        "dataframe": df
    }


def check_extremism_texts(texts: List[str]) -> Dict[str, Any]:
    """Проверка текстов на экстремизм."""
    predictions, probabilities = predict_extremism(texts)

    results = []
    for i, text in enumerate(texts):
        results.append({
            "text": text,
            "prediction": int(predictions[i]),
            "probability": float(probabilities[i]),
            "label": "Экстремистский" if predictions[i] == 1 else "Нейтральный"
        })

    return {
        "total": len(texts),
        "extremist": int((predictions == 1).sum()),
        "neutral": int((predictions == 0).sum()),
        "avg_prob": float(probabilities.mean()),
        "results": results
    }


def get_top_extremist_posts(df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    """Получение топ экстремистских постов."""
    prob_col = "extremism_prob" if "extremism_prob" in df.columns else "probability"
    pred_col = "extremism_pred" if "extremism_pred" in df.columns else "pred"

    if prob_col not in df.columns:
        return []

    top_df = df.nlargest(top_n, prob_col)

    posts = []
    for _, row in top_df.iterrows():
        post = {
            "text": row.get("text", row.get("text_clean", "")),
            "probability": round(float(row[prob_col]), 4),
            "probability_percent": round(float(row[prob_col]) * 100, 2),
            "prediction": int(row.get(pred_col, 1)),
            "label": "Экстремистский" if row.get(pred_col, 1) == 1 else "Нейтральный"
        }

        if "post_id" in row:
            post["post_id"] = row["post_id"]
        if "created_at" in row:
            post["created_at"] = str(row["created_at"])

        posts.append(post)

    return posts


def generate_extremism_visualizations(df: pd.DataFrame, predictions: np.ndarray,
                                       probabilities: np.ndarray) -> Dict[str, str]:
    """Генерация визуализаций для отчета."""
    os.makedirs(STATIC_DIR, exist_ok=True)

    neutral = int((predictions == 0).sum())
    extremist = int((predictions == 1).sum())

    df_copy = df.copy()
    df_copy["extremism_pred"] = predictions
    extremist_texts = df_copy[df_copy["extremism_pred"] == 1]["text_clean"].tolist() \
        if "text_clean" in df_copy.columns else []

    paths = {
        "class_distribution": generate_class_distribution(
            neutral, extremist, filename="extremism_class_distribution.png"
        ),
        "prob_distribution": generate_probability_histogram(
            probabilities, filename="extremism_prob_distribution.png"
        ),
    }

    if extremist_texts:
        paths["wordcloud"] = generate_wordcloud(extremist_texts, filename="extremism_wordcloud.png")
        paths["frequency"] = generate_frequency_chart(extremist_texts, filename="extremism_frequency.png")

    return paths
