# BurstChat

拟人情感陪伴 AI 引擎 — 带 burst 时序控制的群聊框架。

## 核心机制

- **Burst 检测**：用户消息长度决定窗口期（碎片 4s → 长篇 0.5s），模拟真人打字节奏
- **时间戳规划**：LLM 一次性生成带间隔的消息序列 `{"messages": [{"t": 3, "text": "草"}, {"t": 4, "text": "空调坏了？"}]}`
- **动态重规划**：用户插话后自动废弃未发消息，按新上下文重新生成
- **多人设群聊**：每个角色独立调度器，消息互相可见，自然触发回应链

## 快速开始

```bash
pip install openai textual
cp .env.example .env   # 填入 DEEPSEEK_API_KEY=sk-***

# 1v1 聊天
python main.py                          # 小野（默认）
python main.py --persona achen          # 阿辰

# 多人群聊
python demo_group.py                                    # 小野 + 阿辰
python demo_group.py --personas xiaoye achen            # 同上
python demo_group.py --user 咕咕 --personas xiaoye achen  # 自定义你的名字
```

## 项目结构

```
├── main.py              # 1v1 启动入口
├── demo_group.py        # 多人群聊 Demo
├── burstchat/           # 引擎核心
│   ├── prompt.py        # 人设加载 + prompt 模板
│   ├── models.py        # 状态枚举 / 消息结构
│   ├── llm.py           # DeepSeek API 客户端
│   ├── scheduler.py     # 状态机 / burst / 插话 / 重规划
│   └── app.py           # Textual TUI（1v1）
├── personas/            # 人设文件
│   ├── xiaoye.json      # 小野 — 温暖插画师
│   └── achen.json       # 阿辰 — 话痨体育生
├── .env.example
└── requirements.txt
```

## 自定义人设

复制 `personas/xiaoye.json`，修改字段后：

```bash
python main.py --persona 你的角色名
```

无需改任何代码。格式模板、时序规则、输出约束全部自动复用。

## 双人设对比

| | 小野 | 阿辰 |
|---|---|---|
| 人设 | 22岁插画师，养猫 | 20岁体育生，打球 |
| 风格 | 温柔吐槽，颜文字 qwq | 炸裂反应，草/靠/淦 |
| 字数 | ≤12字 | ≤10字 |
| 话量 | 日常2-3条，来劲5条 | 日常3条，刹不住7条 |

## License

MIT
