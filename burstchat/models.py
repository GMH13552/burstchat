"""
数据模型: 状态枚举、消息结构
"""

from dataclasses import dataclass, field


class State:
    IDLE = "idle"
    WAITING_BURST = "waiting_burst"
    PLANNING = "planning"
    DISPATCHING = "dispatching"
    AWAITING_REPLAN = "awaiting_replan"


class PendingMessage:
    __slots__ = ("send_at", "text")

    def __init__(self, send_at: float, text: str):
        self.send_at = send_at  # Unix timestamp
        self.text = text
