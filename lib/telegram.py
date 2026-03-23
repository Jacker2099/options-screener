"""Telegram 发送"""

from __future__ import annotations

import logging
from typing import List

import requests

log = logging.getLogger(__name__)


def send(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        return False
    if len(text) > 4096:
        text = text[:4090] + "\n..."
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            log.warning("TG 失败: %s", resp.text[:200])
        return resp.status_code == 200
    except Exception as e:
        log.warning("TG 异常: %s", e)
        return False


def send_messages(token: str, chat_id: str, messages: List[str]) -> None:
    """按顺序发送多条消息。"""
    import time
    for msg in messages:
        if msg.strip():
            send(token, chat_id, msg)
            time.sleep(0.3)
