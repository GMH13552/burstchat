"""burstchat — 拟人情感陪伴 AI 引擎"""

from .models import State, PendingMessage
from .llm import LLMClient
from .scheduler import Scheduler
from .app import CompanionApp
