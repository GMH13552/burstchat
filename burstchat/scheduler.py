"""
核心调度器: 状态机 + 消息队列 + burst 检测 + 插话重规划
"""

import asyncio
import json
import time
from typing import Optional

from .models import State, PendingMessage
from .llm import LLMClient


class Scheduler:
    def __init__(self, llm: LLMClient, app_callback, name: str = "", json_context: bool = True):
        self.llm = llm
        self.app = app_callback
        self.name = name
        self._on_dispatch: callable = None
        self._on_flush: callable = None
        self._json_context = json_context

        self.state = State.IDLE
        self.context: list[dict] = []
        self.pending: list[PendingMessage] = []
        self.last_user_msg_time: float = 0

        self.burst_timer_task: Optional[asyncio.Task] = None
        self.dispatch_task: Optional[asyncio.Task] = None

        self._pending_texts: list[tuple] = []
        self._interject_at: Optional[float] = None
        self._deferred_user_msgs: list[dict] = []

    # ── Message Entry ───────────────────────────────────────

    async def on_user_message(self, text: str):
        """收到消息 → 进 context → 启动/重置 burst timer"""
        now = time.time()
        self.last_user_msg_time = now
        win = self._burst_window(text)
        self.context.append({"role": "user", "content": text})

        if self.state in (State.DISPATCHING, State.AWAITING_REPLAN):
            self._deferred_user_msgs.append({"role": "user", "content": text})
            self._interject_at = now

        if self.state == State.IDLE:
            self.state = State.WAITING_BURST
            self._start_timer(now, win)
        elif self.state == State.WAITING_BURST:
            self._cancel_timer()
            self._start_timer(now, win)
        elif self.state == State.DISPATCHING:
            self.state = State.AWAITING_REPLAN
            self._cancel_timer()
            self._start_timer(now, win)
            self.app.on_status(f"⏳ 插话检测，{win:.0f}s 后重规划...")
        elif self.state == State.PLANNING:
            self.state = State.WAITING_BURST
            self._cancel_timer()
            self._start_timer(now, win)
        elif self.state == State.AWAITING_REPLAN:
            self._cancel_timer()
            self._start_timer(now, win)

    # ── Burst Window ────────────────────────────────────────

    @staticmethod
    def _burst_window(text: str) -> float:
        n = len(text)
        if n < 5:
            return 4.0
        elif n < 15:
            return 3.0
        elif n < 50:
            return 2.0
        elif n < 100:
            return 1.0
        else:
            return 0.5

    def _start_timer(self, now: float, duration: float):
        self.app.on_status(f"⏳ 等待 {duration:.0f}s...")
        self.burst_timer_task = asyncio.create_task(self._burst_timer(duration))

    def _cancel_timer(self):
        if self.burst_timer_task and not self.burst_timer_task.done():
            self.burst_timer_task.cancel()

    async def _burst_timer(self, duration: float):
        await asyncio.sleep(duration)
        await self._on_burst_end()

    async def _on_burst_end(self):
        now = time.time()

        if self.state == State.WAITING_BURST:
            self.state = State.PLANNING
            self.app.on_status(f"🤔 {self.name} 思考中..." if self.name else "🤔 正在思考...")
            await self._plan_and_dispatch(now, is_replan=False)

        elif self.state == State.AWAITING_REPLAN:
            self.state = State.PLANNING
            self.app.on_status(f"🔄 {self.name} 重新规划..." if self.name else "🔄 重新规划...")
            self._cancel_dispatch()
            self._flush_pending_texts()
            self._clear_remaining_pending()
            await self._plan_and_dispatch(now, is_replan=True)

    # ── Planning ─────────────────────────────────────────────

    async def _plan_and_dispatch(self, now: float, is_replan: bool = False):
        try:
            messages = await self.llm.plan_messages(self.context, now, is_replan=is_replan)

            if self.state != State.PLANNING:
                return

            if not messages:
                self.state = State.IDLE
                self.app.on_status("✅ 就绪")
                return

            self.pending = messages
            self._pending_texts = []
            self.state = State.DISPATCHING

            preview = " → ".join(
                f"[{m.send_at - now:.0f}s] {m.text[:15]}..." for m in messages
            )
            self.app.on_status(f"📤 {self.name} {len(messages)} 条" if self.name else f"📤 {len(messages)} 条")

            self.dispatch_task = asyncio.create_task(self._dispatch_loop())

        except Exception as e:
            self.app.on_status(f"❌ LLM 出错: {e}")
            self.state = State.IDLE

    # ── Dispatch ─────────────────────────────────────────────

    async def _dispatch_loop(self):
        try:
            for msg in self.pending:
                if self.state not in (State.DISPATCHING, State.AWAITING_REPLAN):
                    return

                delay = msg.send_at - time.time()
                while delay > 0:
                    if self.state not in (State.DISPATCHING, State.AWAITING_REPLAN):
                        return
                    await asyncio.sleep(min(delay, 0.2))
                    delay = msg.send_at - time.time()

                if self.state not in (State.DISPATCHING, State.AWAITING_REPLAN):
                    return

                self._dispatch_one(msg)

            if self.state == State.DISPATCHING:
                self._flush_pending_texts()
                if self._on_flush:
                    self._on_flush()
                self.state = State.IDLE
                self.app.on_status("✅ 就绪")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.app.on_status(f"❌ Dispatch 出错: {e}")

    def _dispatch_one(self, msg: PendingMessage):
        now = time.time()
        self.app.on_message(self.name or "assistant", msg.text, now)
        self._pending_texts.append((now, msg.text))
        if self._on_dispatch:
            self._on_dispatch(self.name, msg.text, now)

    # ── Context Flush ────────────────────────────────────────

    def _flush_pending_texts(self):
        if not self._pending_texts and not self._deferred_user_msgs:
            return

        pre, post = [], []
        for send_time, text in self._pending_texts:
            if self._interject_at is not None and send_time > self._interject_at:
                post.append(text)
            else:
                pre.append(text)

        def _fmt(texts):
            return json.dumps({"messages": [{"t": 2 + j * 3, "text": t} for j, t in enumerate(texts)]}, ensure_ascii=False)

        if pre:
            self.context.append({"role": "assistant", "content": _fmt(pre)})
        for m in self._deferred_user_msgs:
            self.context.append(m)
        if post:
            self.context.append({"role": "assistant", "content": _fmt(post)})

        self._pending_texts.clear()
        self._deferred_user_msgs.clear()
        self._interject_at = None

    def _clear_remaining_pending(self):
        count = len(self.pending)
        if count:
            preview = ", ".join(m.text[:10] for m in self.pending)
            self.app.on_status(f"🗑️ 清除了 {count} 条: {preview}...")
        self.pending.clear()

    def _cancel_dispatch(self):
        if self.dispatch_task and not self.dispatch_task.done():
            self.dispatch_task.cancel()
            self.dispatch_task = None

    async def shutdown(self):
        self._cancel_timer()
        self._cancel_dispatch()
