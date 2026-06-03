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
    SEARCHING = "searching"


class PendingMessage:
    __slots__ = ("send_at", "text")

    def __init__(self, send_at: float, text: str):
        self.send_at = send_at  # Unix timestamp
        self.text = text


@dataclass
class PlanResult:
    """LLM 规划结果：消息序列 + 可选的搜索查询"""
    messages: list[PendingMessage] = field(default_factory=list)
    search_query: str = ""  # 非空表示需要在发消息前后执行搜索
