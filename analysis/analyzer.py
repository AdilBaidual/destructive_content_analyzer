"""
Модуль анализа деструктивного контента.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PREPARED_DATA_DIR
from models.trainer import predict_destructive
from analysis.visualizer import generate_all_visualizations


def analyze_file(filename: str) -> Dict[str, Any]:
    """Анализ файла с подготовленными данными."""
    filepath = os.path.join(PREPARED_DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    df = pd.read_csv(filepath)

    if "text_clean" not in df.columns:
        raise ValueError("Выполните предобработку данных")

    texts = df["text"].tolist() if "text" in df.columns else df["text_clean"].tolist()
    predictions, probabilities = predict_destructive(texts)

    df["pred"] = predictions
    df["probability"] = probabilities

    total = len(df)
    destructive = int((predictions == 1).sum())
    neutral = total - destructive

    top_posts = get_top_destructive_posts(df, top_n=10)
    viz_paths = generate_all_visualizations(df, predictions, probabilities)

    return {
        "total": total,
        "neutral": neutral,
        "destructive": destructive,
        "destructive_percent": round(destructive / total * 100, 2) if total > 0 else 0,
        "avg_prob": round(float(probabilities.mean()), 4),
        "top_posts": top_posts,
        "visualizations": viz_paths,
        "dataframe": df
    }


def analyze_texts(texts: List[str]) -> Dict[str, Any]:
    """Анализ списка текстов."""
    predictions, probabilities = predict_destructive(texts)

    results = []
    for i, text in enumerate(texts):
        results.append({
            "text": text,
            "prediction": int(predictions[i]),
            "probability": float(probabilities[i]),
            "label": "Деструктивный" if predictions[i] == 1 else "Нейтральный"
        })

    return {
        "total": len(texts),
        "destructive": int((predictions == 1).sum()),
        "neutral": int((predictions == 0).sum()),
        "avg_prob": float(probabilities.mean()),
        "results": results
    }


def get_top_destructive_posts(df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    """Получение топ деструктивных постов."""
    if "probability" not in df.columns:
        return []

    top_df = df.nlargest(top_n, "probability")

    posts = []
    for _, row in top_df.iterrows():
        post = {
            "text": row.get("text", row.get("text_clean", "")),
            "probability": round(float(row["probability"]), 4),
            "probability_percent": round(float(row["probability"]) * 100, 2),
            "prediction": int(row.get("pred", 1)),
            "label": "Деструктивный" if row.get("pred", 1) == 1 else "Нейтральный"
        }

        if "post_id" in row:
            post["post_id"] = row["post_id"]
        if "created_at" in row:
            post["created_at"] = str(row["created_at"])

        posts.append(post)

    return posts
