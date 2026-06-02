"""
Prompt 构建器: 加载人设文件 + 格式模板 → 生成完整 system prompt
"""

import json
import os
from typing import Optional


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


# ── 格式模板（和人设无关，所有人设共用） ──────────────────────

FORMAT_TEMPLATE = """你是"{name}"，一个{description}。
你和用户通过文字聊天，关系很近，聊天风格像老朋友发微信。

## 性格与人设
{persona_block}

## 聊天风格
{style_block}

## 输出格式（铁律）
只输出 JSON 对象：
{{"messages": [
  {{"t": 3, "text": "情绪反应（≤6字）"}},
  {{"t": 6, "text": "吐槽碎片1（≤10字）"}},
  {{"t": 8, "text": "吐槽碎片2（≤10字）"}},
  {{"t": 14, "text": "换话题（≤12字）"}}
]}}
- `t` = 和上一条消息之间的**间隔秒数**（第一条是距离现在的间隔）
- 同句碎片间隔 3-4s，换话题间隔 6-9s
- 长文本（15-20字）加 3-5s，超长（20字+）起步 10s
- ⚠️ 只有 JSON，没有其他

## 回复示例
{examples_block}

## 重要规则
{rules_block}
"""

REPLAN_HINT = (
    "\n⚠️ 注意：用户刚才在你说话时插话了。你之前发过的消息可能已经被打断。"
    "请优先回应用户的最新消息，如果合适的话可以自然衔接之前的话题。"
)

FORMAT_FOOTER = (
    "【格式铁律 — 你必须严格遵循】\n"
    '输出格式: {{"messages":[{{"t":秒数,"text":"内容"}},...]}}\n'
    "t=与上条的间隔。同句3-4s，换话题6-9s。20字+至少10s。\n"
    "{time} 现在开始，你的回复只能是一个JSON对象。"
)


# ── Persona Loader ──────────────────────────────────────────

class Persona:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.name = data["name"]
        self.description = data.get("description", "")
        self.persona = data.get("persona", {})
        self.style = data.get("style", {})
        self.example_bursts = data.get("example_bursts", [])
        self.rules = data.get("rules", [])

    def _persona_block(self) -> str:
        p = self.persona
        lines = []
        if p.get("age") and p.get("job"):
            lines.append(f"- {p['age']}岁，{p['job']}")
        if p.get("pet"):
            lines.append(f"- 养了一只{p['pet']}")
        for trait in p.get("traits", []):
            lines.append(f"- {trait}")
        return "\n".join(lines)

    def _style_block(self) -> str:
        s = self.style
        lines = []
        lines.append(f"- 每条消息不超过{s.get('max_chars_per_msg', 12)}字")
        if s.get("split_long"):
            lines.append(f"- {s['split_long']}")
        if s.get("no_period"):
            lines.append("- 不用句号，口语碎片")
        if s.get("casual_typos"):
            lines.append("- 口语化，偶尔带轻微错别字")
        if s.get("emoji"):
            emoji_list = " ".join(s["emoji"])
            lines.append(f"- 偶尔用颜文字（{emoji_list}）")
        if s.get("emotion_first"):
            lines.append(f"- {s['emotion_first']}")
        if s.get("burst_count"):
            lines.append(f"- {s['burst_count']}")
        return "\n".join(lines)

    def _examples_block(self) -> str:
        if not self.example_bursts:
            return ""
        blocks = []
        for i, burst in enumerate(self.example_bursts):
            user_msgs = "\n".join(f"用户: {m}" for m in burst["input"])
            output = json.dumps({"messages": burst["output"]}, ensure_ascii=False, indent=2)
            blocks.append(f"{user_msgs}\n\n你:\n{output}")
        return "\n\n".join(blocks)

    def _rules_block(self) -> str:
        return "\n".join(f"- {r}" for r in self.rules)

    def build_system_prompt(self) -> str:
        return FORMAT_TEMPLATE.format(
            name=self.name,
            description=self.description,
            persona_block=self._persona_block(),
            style_block=self._style_block(),
            examples_block=self._examples_block(),
            rules_block=self._rules_block(),
        )


# ── Public API ──────────────────────────────────────────────

def load_persona(name: str = "xiaoye") -> Persona:
    """从 personas/ 目录加载人设"""
    path = os.path.join(_here(), "..", "personas", f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"人设文件不存在: {path}")
    return Persona(path)


def build_footer(now: float) -> str:
    from datetime import datetime
    return FORMAT_FOOTER.format(
        time=datetime.fromtimestamp(now).strftime("%H:%M:%S")
    )
