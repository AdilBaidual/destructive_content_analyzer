"""
Модуль парсинга Telegram-каналов.
"""

import os
import pandas as pd
import datetime
import asyncio
from telethon.sync import TelegramClient
from typing import List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TG_API_ID, TG_API_HASH, RAW_DATA_DIR


def get_client():
    """Создание Telegram клиента."""
    return TelegramClient("session", TG_API_ID, TG_API_HASH)


async def fetch_messages_by_count(channel_name: str, post_count: int) -> str:
    """Парсинг последних N постов канала."""
    async with get_client() as client:
        messages = []
        timestamps = []
        post_ids = []

        async for message in client.iter_messages(channel_name, limit=post_count):
            if message.text:
                messages.append(message.text)
                timestamps.append(message.date)
                post_ids.append(message.id)

        df = pd.DataFrame({
            "text": messages,
            "created_at": timestamps,
            "post_id": post_ids
        })

        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{channel_name.strip('@')}.csv"
        filepath = os.path.join(RAW_DATA_DIR, filename)
        df.to_csv(filepath, index=False)

        return filename


async def fetch_messages_by_ids(channel_name: str, post_ids: List[int]) -> str:
    """Парсинг постов по списку ID."""
    async with get_client() as client:
        entity = await client.get_entity(channel_name)

        messages_data = []
        for post_id in post_ids:
            try:
                message = await client.get_messages(entity, ids=post_id)
                if message and message.text:
                    messages_data.append({
                        "text": message.text,
                        "created_at": message.date,
                        "post_id": message.id
                    })
            except Exception as e:
                print(f"[WARNING] Пост {post_id}: {e}")
                continue

        if not messages_data:
            raise ValueError("Не удалось получить посты")

        df = pd.DataFrame(messages_data)

        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{channel_name.strip('@')}_ids.csv"
        filepath = os.path.join(RAW_DATA_DIR, filename)
        df.to_csv(filepath, index=False)

        return filename


def parse_telegram_channel(channel_name: str, post_count: int) -> str:
    """Парсинг канала по количеству постов."""
    return asyncio.run(fetch_messages_by_count(channel_name, post_count))


def parse_posts_by_ids(channel_name: str, post_ids: List[int]) -> str:
    """Парсинг по списку ID."""
    return asyncio.run(fetch_messages_by_ids(channel_name, post_ids))


def parse_ids_string(ids_string: str) -> List[int]:
    """Парсинг строки ID (например: '123, 456, 789')."""
    ids = []
    for part in ids_string.split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return ids


def load_raw_data(filename: str) -> pd.DataFrame:
    """Загрузка сырых данных."""
    filepath = os.path.join(RAW_DATA_DIR, filename)
    return pd.read_csv(filepath)
