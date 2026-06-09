"""
朋友蒸馏器：读聊天记录 → 统计发消息规律 → LLM 总结人设
输出 burstchat 兼容的 persona JSON

用法：
  python distill.py chat.csv --speaker-col 发言者 --time-col 时间 --text-col 消息
  python distill.py chat.csv -s 发言者 -t 时间 -m 消息 --target "张三"
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from statistics import mean, median, stdev

import httpx  # 用 httpx 异步调 DeepSeek API
from bs4 import BeautifulSoup


# ═══════════════════════════════════════════════════════════════
# 统计工具
# ═══════════════════════════════════════════════════════════════

def parse_csv(path, speaker_col, time_col, text_col, time_fmt=None):
    """读取聊天 CSV，返回 [{speaker, time, text}, ...]"""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = row[time_col].strip()
                if time_fmt:
                    t = datetime.strptime(ts, time_fmt)
                else:
                    t = _auto_parse_time(ts)
                rows.append({
                    "speaker": row[speaker_col].strip(),
                    "time": t,
                    "text": row[text_col].strip(),
                })
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r["time"])
    return rows


def _auto_parse_time(s):
    """尝试常见时间格式"""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # 最后尝试 ISO
    return datetime.fromisoformat(s)


# ═══════════════════════════════════════════════════════════════
# 特征提取
# ═══════════════════════════════════════════════════════════════

def compute_stats(messages):
    """
    从一个发言者的消息列表（按时序排列）中提取特征。
    返回 dict，所有数值保留 2 位小数。
    """
    if not messages:
        return {}

    texts = [m["text"] for m in messages]
    times = [m["time"] for m in messages]
    lengths = [len(t) for t in texts]

    # ── 基础 ──
    total = len(messages)
    avg_len = round(mean(lengths), 1)

    # ── Burst 检测：连续发几条才停 ──
    bursts = _detect_bursts(messages)
    burst_sizes = [len(b) for b in bursts]
    # "正常 burst" = 排除单条（可能是回复别人的一句）
    if burst_sizes:
        avg_burst = round(mean(burst_sizes), 1)
    else:
        avg_burst = 1.0

    # ── Burst 内间隔（同一个人连发时，条与条之间的间隔） ──
    intra_gaps = []
    for burst in bursts:
        for i in range(1, len(burst)):
            gap = (burst[i]["time"] - burst[i - 1]["time"]).total_seconds()
            if gap <= 60:  # 60s 内的才算 burst 内连发
                intra_gaps.append(gap)
    avg_intra_gap = round(mean(intra_gaps), 1) if intra_gaps else 3.0

    # ── 消息长度分布 ──
    short_pct = round(sum(1 for l in lengths if l <= 6) / total * 100, 1)
    mid_pct = round(sum(1 for l in lengths if 7 <= l <= 15) / total * 100, 1)
    long_pct = round(sum(1 for l in lengths if l >= 16) / total * 100, 1)

    # ── Emoji ──
    # 只匹配真正的 emoji 和颜文字，不碰 CJK 范围
    emoji_re = re.compile(
        r'[\U0001F600-\U0001F64F]'       # 表情符号
        r'|[\U0001F300-\U0001F5FF]'       # 杂项符号/象形
        r'|[\U0001F680-\U0001F6FF]'       # 交通/地图
        r'|[\U0001F1E0-\U0001F1FF]'       # 国旗
        r'|[\U0001F900-\U0001F9FF]'       # 补充符号
        r'|[\U0001FA00-\U0001FA6F]'       # 象棋/扩展
        r'|[\U0001FA70-\U0001FAFF]'       # 扩展-A
        r'|[\u2600-\u26FF]'               # 杂项符号（☀☁☂★等）
        r'|[\u2700-\u27BF]'               # 装饰符号（✂✈✉等）
        r'|[❤💕💔💖💗💙💚💛💜💝💞💟❣💌💋💯🔥⭐✨🌟💥💦💨💫🕊☠💀👀🧠🫀🫁]'  # 常用
        r'|[qwQqwpTATOrzZ]{2,6}'            # 颜文字核心
        r'|[👉👈🫡🥺🫠🤡🤯🫣😭😅😁😂🤣😊🙏🥰😍😒😢👍😡🤔🫵🤗🫰🤌]'  # 常用手势表情
    )
    emoji_count = sum(len(emoji_re.findall(t)) for t in texts)
    emoji_per_msg = round(emoji_count / total, 2)

    # ── 活跃时段 ──
    hour_counts = Counter()
    for t in times:
        hour_counts[t.hour] += 1
    top_hours = sorted(hour_counts.most_common(6))
    active_windows = []
    if top_hours:
        ranges = []
        start = top_hours[0][0]
        end = top_hours[0][0]
        for i in range(1, len(top_hours)):
            if top_hours[i][0] == end + 1:
                end = top_hours[i][0]
            else:
                ranges.append((start, end))
                start = end = top_hours[i][0]
        ranges.append((start, end))
        active_windows = [f"{s}:00-{e}:00" for s, e in ranges]

    # ── 常用词 ──
    all_words = []
    for t in texts:
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', t)
        all_words.extend(words)
    word_freq = Counter(all_words).most_common(20)
    top_words = [w for w, c in word_freq if c >= 3]

    return {
        "total_msgs": total,
        "avg_len": avg_len,
        "short_pct": short_pct,
        "mid_pct": mid_pct,
        "long_pct": long_pct,
        "avg_burst_size": avg_burst,
        "avg_intra_gap_sec": avg_intra_gap,
        "emoji_per_msg": emoji_per_msg,
        "active_hours": active_windows,
        "top_words": top_words[:10],
    }


def _detect_bursts(messages, gap_threshold=180):
    """把消息按时间间隔拆成 burst 组（间隔 >180s 算新 burst）"""
    if not messages:
        return []
    bursts = []
    current = [messages[0]]
    for i in range(1, len(messages)):
        gap = (messages[i]["time"] - messages[i - 1]["time"]).total_seconds()
        if gap > gap_threshold:
            bursts.append(current)
            current = [messages[i]]
        else:
            current.append(messages[i])
    if current:
        bursts.append(current)
    return bursts


# ═══════════════════════════════════════════════════════════════
# 回复延迟分析（更拟人的关键）
# ═══════════════════════════════════════════════════════════════

def compute_reply_delays(all_messages, target_speaker):
    """
    分析 target_speaker 的回复延迟：
    - 从别人发完消息到 target 回复，间隔多少秒
    - 按消息长度分桶（短消息秒回，长消息慢一点）
    - 按小时段分（半夜可能不回）
    """
    delays = []
    last_other_time = None

    for m in all_messages:
        if m["speaker"] != target_speaker:
            last_other_time = m["time"]
        elif last_other_time is not None:
            delay = (m["time"] - last_other_time).total_seconds()
            if delay >= 0 and delay < 86400:  # 忽略超长间隔（换了话题）
                delays.append({
                    "delay_sec": delay,
                    "hour": m["time"].hour,
                    "text_len": len(m["text"]),
                })
            last_other_time = None

    if not delays:
        return {}

    all_delays = [d["delay_sec"] for d in delays]
    # 快回（<60s）
    instant_pct = round(sum(1 for d in all_delays if d < 60) / len(all_delays) * 100, 1)
    # 中速（60-600s）
    medium_pct = round(sum(1 for d in all_delays if 60 <= d < 600) / len(all_delays) * 100, 1)
    # 慢回（≥600s）
    slow_pct = round(sum(1 for d in all_delays if d >= 600) / len(all_delays) * 100, 1)

    # 按小时分组
    hour_delays = defaultdict(list)
    for d in delays:
        hour_delays[d["hour"]].append(d["delay_sec"])
    slowest_hours = sorted(
        [(h, mean(v)) for h, v in hour_delays.items()],
        key=lambda x: x[1], reverse=True
    )[:3]

    # 延迟和消息长度的关系
    short_reply_delays = [d["delay_sec"] for d in delays if d["text_len"] <= 6]
    long_reply_delays = [d["delay_sec"] for d in delays if d["text_len"] >= 20]
    short_avg = round(mean(short_reply_delays), 1) if short_reply_delays else None
    long_avg = round(mean(long_reply_delays), 1) if long_reply_delays else None

    return {
        "avg_delay_sec": round(mean(all_delays), 1),
        "median_delay_sec": round(median(all_delays), 1),
        "instant_pct": instant_pct,  # <60s
        "medium_pct": medium_pct,  # 60-600s
        "slow_pct": slow_pct,  # ≥600s
        "slowest_hours": [(h, f"{round(v, 0)}s") for h, v in slowest_hours],
        "short_msg_reply_avg": short_avg,
        "long_msg_reply_avg": long_avg,
    }


# ═══════════════════════════════════════════════════════════════
# LLM 蒸馏
# ═══════════════════════════════════════════════════════════════

DISTILL_PROMPT = """你是一个人物画像分析师。根据以下聊天数据，为 {name} 生成一份精确的人设描述。

## 数据指标
- 总消息数: {total_msgs}
- 平均每条 {avg_len} 字
- 短消息(≤6字)占比 {short_pct}%，中等(7-15字)占比 {mid_pct}%，长消息(≥16字)占比 {long_pct}%
- 一次发 {avg_burst_size} 条（平均 burst 大小）
- 条间间隔 {avg_intra_gap_sec}s
- 每条 emoji 数 {emoji_per_msg}
- 活跃时段: {active_hours}
- 常用词: {top_words}

## 回复延迟
{delay_stats}

## 消息样本
{samples}

## 任务
根据以上数据，输出一个 burstchat 人设 JSON。要求：
- name: 一个合适的昵称（2-3个字）
- description: 一句话描述这个人的性格
- persona: age, job（猜一个合适的）, pet（没有就 null）, traits（3-5个性格特点）
- style: 
  - max_chars_per_msg: 根据消息长度分布定（短消息多就 ≤10，中等多就 12，长消息多就 15+）
  - emoji: 列出 ta 常用的 emoji（2-4个）
  - split_long: 长句拆分策略
  - emotion_first: 情绪表达策略
  - burst_count: 根据 burst 大小描述
- example_bursts: 根据样本生成 3 个逼真的对话示例（每个 2-4 条回复，带合理的 t 间隔）
- rules: 3-4 条行为规则

输出纯 JSON，不要 markdown 包裹。"""


async def distill(name, stats, delays, samples, api_key, model="deepseek-chat"):
    """用 LLM 蒸馏人设"""
    async with httpx.AsyncClient(timeout=60) as client:
        prompt_content = DISTILL_PROMPT.format(
                        name=name,
                        total_msgs=stats.get("total_msgs", 0),
                        avg_len=stats.get("avg_len", 0),
                        short_pct=stats.get("short_pct", 0),
                        mid_pct=stats.get("mid_pct", 0),
                        long_pct=stats.get("long_pct", 0),
                        avg_burst_size=stats.get("avg_burst_size", 1),
                        avg_intra_gap_sec=stats.get("avg_intra_gap_sec", 3),
                        emoji_per_msg=stats.get("emoji_per_msg", 0),
                        active_hours=", ".join(stats.get("active_hours", [])),
                        top_words=", ".join(stats.get("top_words", [])),
                        delay_stats=json.dumps(delays, ensure_ascii=True),
                        samples=samples,
                    )
        # httpx json serializer chokes on non-ASCII; force-safe it
        prompt_content = prompt_content.encode('utf-8').decode('utf-8')

        resp = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt_content},
                ],
                "temperature": 0.7,
                "max_tokens": 2000,
            },
        )
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.rstrip().endswith("```"):
                content = content.rsplit("\n", 1)[0]
        return json.loads(content)


# ═══════════════════════════════════════════════════════════════
# 生成样本文本
# ═══════════════════════════════════════════════════════════════

def format_samples(messages, max_msgs=30):
    """选取代表性样本（首、中、尾各取一些）"""
    if len(messages) <= max_msgs:
        selected = messages
    else:
        n = len(messages)
        indices = (
            list(range(0, max_msgs // 3)) +
            list(range(n // 2 - max_msgs // 6, n // 2 + max_msgs // 6)) +
            list(range(n - max_msgs // 3, n))
        )
        selected = [messages[i] for i in sorted(set(indices)) if i < n]

    lines = []
    for m in selected[:max_msgs]:
        t = m["time"].strftime("%m/%d %H:%M")
        clean_text = m['text'].replace('…', '...').replace('⋯', '...')
        # ASCII-safe for httpx
        try:
            clean_text.encode('ascii')
        except UnicodeEncodeError:
            clean_text = clean_text.encode('ascii', errors='replace').decode('ascii')
        lines.append(f"[{t}] {clean_text}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

async def main():
    # Windows 终端编码修复
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="朋友蒸馏器 — 聊天记录 → persona JSON")
    parser.add_argument("file", help="聊天记录 CSV 文件")
    parser.add_argument("-s", "--speaker-col", default="speaker", help="发言者列名")
    parser.add_argument("-t", "--time-col", default="time", help="时间列名")
    parser.add_argument("-m", "--text-col", default="text", help="消息内容列名")
    parser.add_argument("--time-fmt", help="时间格式（自动检测）")
    parser.add_argument("--target", help="只分析指定对象")
    parser.add_argument("-o", "--output", default="personas", help="输出目录")
    parser.add_argument("--api-key", help="DeepSeek API Key（默认用环境变量 DEEPSEEK_KEY）")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--no-llm", action="store_true", help="只统计，不调 LLM")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_KEY")
    if not api_key and not args.no_llm:
        print("❌ 请设置 DEEPSEEK_KEY 环境变量或用 --api-key")
        sys.exit(1)

    # ── 读取 ──
    print(f"📖 读取 {args.file} ...")
    rows = parse_csv(args.file, args.speaker_col, args.time_col, args.text_col, args.time_fmt)
    speakers = sorted(set(r["speaker"] for r in rows))
    print(f"   共 {len(rows)} 条消息，{len(speakers)} 个人: {', '.join(speakers)}")

    # ── 分组 ──
    by_speaker = defaultdict(list)
    for r in rows:
        by_speaker[r["speaker"]].append(r)

    # 时间范围
    all_times = [r["time"] for r in rows]
    print(f"   时间范围: {min(all_times)} ~ {max(all_times)}")

    targets = [args.target] if args.target else speakers

    # ── 输出目录 ──
    os.makedirs(args.output, exist_ok=True)

    for name in targets:
        msgs = by_speaker.get(name, [])
        if not msgs:
            print(f"\n⚠️  没找到 {name} 的消息，跳过")
            continue

        print(f"\n{'='*50}")
        print(f"🔍 分析: {name} ({len(msgs)} 条消息)")

        # 统计
        stats = compute_stats(msgs)
        delays = compute_reply_delays(rows, name)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print("回复延迟:", json.dumps(delays, ensure_ascii=False, indent=2))

        if args.no_llm:
            output_path = os.path.join(args.output, f"{name}_stats.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"stats": stats, "delays": delays}, f, ensure_ascii=False, indent=2)
            print(f"📊 统计已保存: {output_path}")
            continue

        # LLM 蒸馏
        samples = format_samples(msgs)
        print(f"🤖 蒸馏中 ...")
        try:
            persona = await distill(name, stats, delays, samples, api_key, args.model)
            persona["reply_profile"] = {
                "avg_delay_sec": delays.get("avg_delay_sec"),
                "median_delay_sec": delays.get("median_delay_sec"),
                "instant_pct": delays.get("instant_pct"),
                "medium_pct": delays.get("medium_pct"),
                "slow_pct": delays.get("slow_pct"),
                "short_msg_reply_avg": delays.get("short_msg_reply_avg"),
                "long_msg_reply_avg": delays.get("long_msg_reply_avg"),
            }
            persona["_source"] = {
                "total_msgs": stats["total_msgs"],
                "date_range": f"{min(all_times)} ~ {max(all_times)}",
                "avg_reply_delay_sec": delays.get("avg_delay_sec"),
                "instant_reply_pct": delays.get("instant_pct"),
            }
            output_path = os.path.join(args.output, f"{persona.get('name', name)}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(persona, f, ensure_ascii=False, indent=2)
            print(f"✅ 已保存: {output_path}")
            print(f"   {persona.get('name')}: {persona.get('description')}")
        except Exception as e:
            print(f"❌ 蒸馏失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
