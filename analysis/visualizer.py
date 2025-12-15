"""
Модуль визуализации результатов анализа.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import re
from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STATIC_DIR


def ensure_static_dir():
    os.makedirs(STATIC_DIR, exist_ok=True)


def generate_class_distribution(neutral: int, destructive: int,
                                filename: str = "class_distribution.png") -> str:
    """График распределения по классам."""
    ensure_static_dir()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Нейтральный", "Деструктивный"], [neutral, destructive],
                  color=["#2ecc71", "#e74c3c"])

    for bar, val in zip(bars, [neutral, destructive]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=12)

    ax.set_title("Распределение по классам", fontsize=14)
    ax.set_ylabel("Количество постов", fontsize=12)

    filepath = os.path.join(STATIC_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close(fig)

    return filepath


def generate_probability_histogram(probabilities: np.ndarray,
                                   filename: str = "prob_distribution.png") -> str:
    """Гистограмма распределения вероятностей."""
    ensure_static_dir()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(probabilities, bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', label='Порог 0.5')
    ax.set_title("Распределение вероятностей", fontsize=14)
    ax.set_xlabel("Вероятность", fontsize=12)
    ax.set_ylabel("Количество", fontsize=12)
    ax.legend()

    filepath = os.path.join(STATIC_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close(fig)

    return filepath


def generate_timeline(df: pd.DataFrame, prob_column: str = "probability",
                      date_column: str = "created_at", filename: str = "timeline.png") -> str:
    """График временной динамики."""
    ensure_static_dir()

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df["date"] = df[date_column].dt.date

    timeline = df.groupby("date")[prob_column].agg(['mean', 'count']).reset_index()
    timeline.columns = ['date', 'avg_prob', 'count']

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color1 = '#e74c3c'
    ax1.plot(timeline["date"], timeline["avg_prob"], marker='o',
             linestyle='-', color=color1, linewidth=2, markersize=6)
    ax1.set_xlabel("Дата", fontsize=12)
    ax1.set_ylabel("Средняя вероятность", color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    color2 = '#3498db'
    ax2.bar(timeline["date"], timeline["count"], alpha=0.3, color=color2)
    ax2.set_ylabel("Количество постов", color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title("Динамика по датам", fontsize=14)
    plt.xticks(rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.5)

    filepath = os.path.join(STATIC_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close(fig)

    return filepath


def generate_wordcloud(texts: list, filename: str = "wordcloud.png") -> str:
    """Облако слов."""
    ensure_static_dir()

    combined_text = " ".join(str(t) for t in texts if t)

    if not combined_text.strip():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Нет данных", ha='center', va='center', fontsize=14)
        ax.axis('off')
        filepath = os.path.join(STATIC_DIR, filename)
        plt.savefig(filepath, dpi=100)
        plt.close(fig)
        return filepath

    try:
        russian_stopwords = set(stopwords.words("russian"))
    except:
        russian_stopwords = set()

    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        stopwords=russian_stopwords,
        max_words=100,
        colormap='Reds'
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Облако слов", fontsize=14)

    filepath = os.path.join(STATIC_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close(fig)

    return filepath


def generate_frequency_chart(texts: list, top_n: int = 10,
                             filename: str = "frequency.png") -> str:
    """График частотности слов."""
    ensure_static_dir()

    combined_text = " ".join(str(t) for t in texts if t)
    words = re.findall(r"\w{3,}", combined_text.lower())

    try:
        russian_stopwords = set(stopwords.words("russian"))
    except:
        russian_stopwords = set()

    words = [w for w in words if w not in russian_stopwords]

    if not words:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Нет данных", ha='center', va='center', fontsize=14)
        ax.axis('off')
        filepath = os.path.join(STATIC_DIR, filename)
        plt.savefig(filepath, dpi=100)
        plt.close(fig)
        return filepath

    word_counts = Counter(words).most_common(top_n)
    words_list = [w for w, _ in word_counts]
    counts_list = [c for _, c in word_counts]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(words_list[::-1], counts_list[::-1], color='#3498db')
    ax.set_xlabel("Частота", fontsize=12)
    ax.set_title(f"Топ-{top_n} слов", fontsize=14)

    filepath = os.path.join(STATIC_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close(fig)

    return filepath


def generate_top_posts_chart(df: pd.DataFrame, top_n: int = 10,
                             prob_column: str = "probability",
                             filename: str = "top_posts.png") -> str:
    """График топ постов по вероятности."""
    ensure_static_dir()

    top_df = df.nlargest(top_n, prob_column)

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    for idx, row in top_df.iterrows():
        post_id = row.get("post_id", idx)
        labels.append(f"#{post_id}")

    colors = plt.cm.Reds(top_df[prob_column].values)
    ax.barh(range(len(top_df)), top_df[prob_column].values, color=colors)

    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Вероятность", fontsize=12)
    ax.set_title(f"Топ-{top_n} постов", fontsize=14)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    filepath = os.path.join(STATIC_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close(fig)

    return filepath


def generate_all_visualizations(df: pd.DataFrame, predictions: np.ndarray,
                                probabilities: np.ndarray) -> dict:
    """Генерация всех визуализаций."""
    df = df.copy()
    df["pred"] = predictions
    df["probability"] = probabilities

    neutral = int((predictions == 0).sum())
    destructive = int((predictions == 1).sum())

    destructive_texts = df[df["pred"] == 1]["text_clean"].tolist() \
        if "text_clean" in df.columns else []

    paths = {
        "class_distribution": generate_class_distribution(neutral, destructive),
        "prob_distribution": generate_probability_histogram(probabilities),
        "wordcloud": generate_wordcloud(destructive_texts),
        "frequency": generate_frequency_chart(destructive_texts),
        "top_posts": generate_top_posts_chart(df)
    }

    if "created_at" in df.columns:
        paths["timeline"] = generate_timeline(df)

    return paths
