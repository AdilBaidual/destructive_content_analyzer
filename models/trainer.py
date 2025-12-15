"""
Модуль обучения и дообучения ML моделей.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TOXIC_COMMENTS_PATH, EXTREMISM_DATA_PATH,
    ADDITIONAL_DESTRUCTIVE_PATH, ADDITIONAL_EXTREMISM_PATH,
    DESTRUCTIVE_MODEL_DIR, DESTRUCTIVE_MODEL_PATH, DESTRUCTIVE_VECTORIZER_PATH,
    EXTREMISM_MODEL_DIR, EXTREMISM_MODEL_PATH, EXTREMISM_TOKENIZER_PATH, EXTREMISM_MAXLEN_PATH,
    TFIDF_MAX_FEATURES, KERAS_VOCAB_SIZE, KERAS_EMBEDDING_DIM, KERAS_MAXLEN_DEFAULT
)
from preprocessing.preprocessor import preprocess_text


# =============================================================================
# МОДЕЛЬ ДЕСТРУКТИВА (SGDClassifier + TF-IDF)
# =============================================================================

def load_destructive_dataset() -> pd.DataFrame:
    """Загрузка датасета с учетом дополнительных данных для дообучения."""
    df_base = pd.read_csv(TOXIC_COMMENTS_PATH)

    if os.path.exists(ADDITIONAL_DESTRUCTIVE_PATH):
        df_add = pd.read_csv(ADDITIONAL_DESTRUCTIVE_PATH)
        df = pd.concat([df_base, df_add], ignore_index=True)
        print(f"[INFO] Объединено {len(df_base)} + {len(df_add)} = {len(df)} примеров")
    else:
        df = df_base
        print(f"[INFO] Загружено {len(df)} примеров")

    return df


def train_destructive_model(test_size: float = 0.2) -> Dict[str, Any]:
    """Обучение модели деструктива на датасете."""
    print("[INFO] Обучение модели деструктива...")

    df = load_destructive_dataset()
    df = df.dropna(subset=["text", "label"])

    if "text_clean" not in df.columns:
        df["text_clean"] = df["text"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"], df["label"],
        test_size=test_size, random_state=42
    )

    # TF-IDF векторизация текстов
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # SGDClassifier с log_loss эквивалентен логистической регрессии,
    # но поддерживает инкрементальное обучение (partial_fit)
    model = SGDClassifier(
        loss='log_loss',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[INFO] Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(DESTRUCTIVE_MODEL_DIR, exist_ok=True)
    joblib.dump(model, DESTRUCTIVE_MODEL_PATH)
    joblib.dump(vectorizer, DESTRUCTIVE_VECTORIZER_PATH)

    return {
        "accuracy": accuracy,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }


def load_destructive_model() -> Tuple[SGDClassifier, TfidfVectorizer]:
    """Загрузка обученной модели."""
    if not os.path.exists(DESTRUCTIVE_MODEL_PATH):
        raise FileNotFoundError("Модель не обучена")

    model = joblib.load(DESTRUCTIVE_MODEL_PATH)
    vectorizer = joblib.load(DESTRUCTIVE_VECTORIZER_PATH)
    return model, vectorizer


def retrain_destructive_model(new_texts: list, new_labels: list) -> Dict[str, Any]:
    """Дообучение модели на новых данных через partial_fit."""
    print(f"[INFO] Дообучение на {len(new_texts)} примерах...")

    model, vectorizer = load_destructive_model()
    new_texts_clean = [preprocess_text(t) for t in new_texts]
    X_new = vectorizer.transform(new_texts_clean)

    # Инкрементальное обучение без полного переобучения
    model.partial_fit(X_new, new_labels)

    joblib.dump(model, DESTRUCTIVE_MODEL_PATH)
    add_to_additional_dataset(new_texts, new_labels, ADDITIONAL_DESTRUCTIVE_PATH)

    return {"status": "success", "new_examples": len(new_texts)}


def predict_destructive(texts: list) -> Tuple[np.ndarray, np.ndarray]:
    """Предсказание деструктивности для текстов."""
    model, vectorizer = load_destructive_model()

    texts_clean = [preprocess_text(t) for t in texts]
    X = vectorizer.transform(texts_clean)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities


# =============================================================================
# МОДЕЛЬ ЭКСТРЕМИЗМА (Keras Sequential)
# =============================================================================

def load_extremism_dataset() -> pd.DataFrame:
    """Загрузка датасета экстремизма."""
    df_base = pd.read_csv(EXTREMISM_DATA_PATH)

    if os.path.exists(ADDITIONAL_EXTREMISM_PATH):
        df_add = pd.read_csv(ADDITIONAL_EXTREMISM_PATH)
        df = pd.concat([df_base, df_add], ignore_index=True)
        print(f"[INFO] Объединено {len(df_base)} + {len(df_add)} = {len(df)} примеров")
    else:
        df = df_base
        print(f"[INFO] Загружено {len(df)} примеров")

    return df


def save_maxlen(maxlen: int):
    """Сохранение максимальной длины последовательности."""
    os.makedirs(EXTREMISM_MODEL_DIR, exist_ok=True)
    with open(EXTREMISM_MAXLEN_PATH, "w") as f:
        f.write(str(maxlen))


def load_maxlen() -> int:
    """Загрузка maxlen."""
    if not os.path.exists(EXTREMISM_MAXLEN_PATH):
        return KERAS_MAXLEN_DEFAULT
    with open(EXTREMISM_MAXLEN_PATH, "r") as f:
        return int(f.read().strip())


def save_tokenizer(tokenizer: Tokenizer):
    """Сохранение токенизатора в JSON."""
    os.makedirs(EXTREMISM_MODEL_DIR, exist_ok=True)
    with open(EXTREMISM_TOKENIZER_PATH, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())


def load_tokenizer() -> Tokenizer:
    """Загрузка токенизатора."""
    with open(EXTREMISM_TOKENIZER_PATH, "r", encoding="utf-8") as f:
        return tokenizer_from_json(f.read())


def create_keras_model(vocab_size: int = KERAS_VOCAB_SIZE) -> Sequential:
    """
    Создание нейросетевой модели.

    Архитектура:
    - Embedding: преобразование слов в векторы
    - GlobalAveragePooling1D: усреднение векторов слов
    - Dense(16, relu): скрытый слой
    - Dense(1, sigmoid): выход - вероятность экстремизма
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=KERAS_EMBEDDING_DIM),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_extremism_model(epochs: int = 10, test_size: float = 0.2) -> Dict[str, Any]:
    """Обучение нейросетевой модели экстремизма."""
    print("[INFO] Обучение модели экстремизма...")

    df = load_extremism_dataset()
    df = df.dropna(subset=["text", "label"])

    # Определение максимальной длины текста (без предобработки, как в оригинале)
    max_text_len = df["text"].apply(lambda x: len(str(x).split())).max()
    maxlen = min(max_text_len, 100)

    # Токенизация: преобразование текста в последовательность индексов
    # Без предобработки - как в оригинальном extremism_detector
    tokenizer = Tokenizer(num_words=KERAS_VOCAB_SIZE)
    tokenizer.fit_on_texts(df["text"])

    X_seq = tokenizer.texts_to_sequences(df["text"])
    X_pad = pad_sequences(X_seq, maxlen=maxlen)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=test_size, random_state=42
    )

    model = create_keras_model(vocab_size=KERAS_VOCAB_SIZE)
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"[INFO] Test Accuracy: {accuracy:.4f}")

    os.makedirs(EXTREMISM_MODEL_DIR, exist_ok=True)
    model.save(EXTREMISM_MODEL_PATH)
    save_tokenizer(tokenizer)
    save_maxlen(maxlen)

    return {
        "accuracy": float(accuracy),
        "loss": float(loss),
        "epochs": epochs,
        "maxlen": maxlen
    }


def load_extremism_model() -> Tuple[Sequential, Tokenizer, int]:
    """Загрузка обученной модели."""
    if not os.path.exists(EXTREMISM_MODEL_PATH):
        raise FileNotFoundError("Модель не обучена")

    model = load_model(EXTREMISM_MODEL_PATH)
    tokenizer = load_tokenizer()
    maxlen = load_maxlen()

    return model, tokenizer, maxlen


def retrain_extremism_model(new_texts: list, new_labels: list, epochs: int = 5) -> Dict[str, Any]:
    """Дообучение модели на новых данных."""
    print(f"[INFO] Дообучение на {len(new_texts)} примерах...")

    model, tokenizer, maxlen = load_extremism_model()

    # Если новые тексты длиннее maxlen - полное переобучение
    new_max_len = max(len(t.split()) for t in new_texts)
    if new_max_len > maxlen:
        print("[WARNING] Тексты длиннее maxlen. Переобучение...")
        add_to_additional_dataset(new_texts, new_labels, ADDITIONAL_EXTREMISM_PATH)
        return train_extremism_model(epochs=10)

    # Без предобработки - как в оригинальном extremism_detector
    X_new = tokenizer.texts_to_sequences(new_texts)
    X_new_pad = pad_sequences(X_new, maxlen=maxlen)
    y_new = np.array(new_labels)

    history = model.fit(X_new_pad, y_new, epochs=epochs, verbose=1)
    model.save(EXTREMISM_MODEL_PATH)

    add_to_additional_dataset(new_texts, new_labels, ADDITIONAL_EXTREMISM_PATH)

    return {
        "status": "success",
        "new_examples": len(new_texts),
        "final_accuracy": float(history.history['accuracy'][-1])
    }


def predict_extremism(texts: list) -> Tuple[np.ndarray, np.ndarray]:
    """Предсказание экстремизма для текстов."""
    model, tokenizer, maxlen = load_extremism_model()

    # Без предобработки - как в оригинальном extremism_detector
    X_seq = tokenizer.texts_to_sequences(texts)
    X_pad = pad_sequences(X_seq, maxlen=maxlen)

    probabilities = model.predict(X_pad, verbose=0).flatten()
    predictions = (probabilities >= 0.5).astype(int)

    return predictions, probabilities


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def add_to_additional_dataset(texts: list, labels: list, filepath: str):
    """Сохранение новых примеров в дополнительный датасет."""
    new_df = pd.DataFrame({"text": texts, "label": labels})

    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(filepath, index=False)


def train_all_models():
    """Обучение обеих моделей."""
    print("=" * 50)
    print("Обучение модели деструктива")
    print("=" * 50)
    destructive_metrics = train_destructive_model()

    print("\n" + "=" * 50)
    print("Обучение модели экстремизма")
    print("=" * 50)
    extremism_metrics = train_extremism_model()

    return {"destructive": destructive_metrics, "extremism": extremism_metrics}


def check_models_exist() -> Dict[str, bool]:
    """Проверка наличия обученных моделей."""
    return {
        "destructive": os.path.exists(DESTRUCTIVE_MODEL_PATH),
        "extremism": os.path.exists(EXTREMISM_MODEL_PATH)
    }


if __name__ == "__main__":
    train_all_models()
