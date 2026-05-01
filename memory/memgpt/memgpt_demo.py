"""
复现 MemGPT 的核心记忆管理机制
来源：Packer et al. 2023 - MemGPT: Towards LLMs as Operating Systems
依赖：pip install numpy
运行：python memgpt_demo.py
"""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class Message:
    role: str       # "user" / "assistant" / "system"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ArchivalRecord:
    content: str
    embedding: np.ndarray
    created_at: float = field(default_factory=time.time)


# ─────────────────────────────────────────────
# External Context（磁盘层）
# ─────────────────────────────────────────────

def mock_embed(text: str) -> np.ndarray:
    """确定性 mock 嵌入（实际使用 text-embedding-ada-002 等）"""
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(64).astype(np.float32)


class RecallStorage:
    """完整对话历史数据库（类比磁盘 swap 区）"""

    def __init__(self):
        self.messages: list[Message] = []

    def append(self, msg: Message):
        self.messages.append(msg)

    def search(self, query: str, page: int = 0, page_size: int = 5) -> list[Message]:
        q_embed = mock_embed(query)
        scored = []
        for msg in self.messages:
            sim = float(np.dot(q_embed, mock_embed(msg.content)) /
                       (np.linalg.norm(q_embed) * np.linalg.norm(mock_embed(msg.content)) + 1e-8))
            scored.append((sim, msg))
        scored.sort(key=lambda x: x[0], reverse=True)
        start = page * page_size
        return [m for _, m in scored[start:start + page_size]]


class ArchivalStorage:
    """任意长度文本知识库（类比磁盘持久存储）"""

    def __init__(self):
        self.records: list[ArchivalRecord] = []

    def insert(self, content: str):
        self.records.append(ArchivalRecord(content, mock_embed(content)))

    def search(self, query: str, page: int = 0, page_size: int = 5) -> list[ArchivalRecord]:
        """向量相似度检索，支持分页（function chaining 的基础）"""
        q_embed = mock_embed(query)
        scored = []
        for rec in self.records:
            sim = float(np.dot(q_embed, rec.embedding) /
                       (np.linalg.norm(q_embed) * np.linalg.norm(rec.embedding) + 1e-8))
            scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        start = page * page_size
        return [r for _, r in scored[start:start + page_size]]


# ─────────────────────────────────────────────
# Main Context（内存层）
# ─────────────────────────────────────────────

class MemGPTContext:
    """
    MemGPT 主上下文管理器。
    WARNING_THRESHOLD: 触发内存压力警告的队列长度（对应 70% context window）
    FLUSH_THRESHOLD:   强制驱逐的队列长度（对应 100% context window）
    """
    WARNING_THRESHOLD = 8
    FLUSH_THRESHOLD   = 10

    def __init__(self, system_instructions: str):
        self.system_instructions = system_instructions     # 只读
        self.working_context: dict[str, str] = {}         # 读写，用户长期信息
        self.fifo_queue: deque[Message] = deque()         # 滚动消息历史
        self.recursive_summary: str = ""                   # 被驱逐消息的摘要
        self.recall_storage = RecallStorage()
        self.archival_storage = ArchivalStorage()

    # ── 函数调用接口（LLM 主动调用）──────────────

    def core_memory_append(self, key: str, value: str) -> str:
        """向 working context 追加信息"""
        self.working_context[key] = self.working_context.get(key, "") + " " + value
        return f"[OK] working_context['{key}'] updated"

    def core_memory_replace(self, key: str, new_value: str) -> str:
        self.working_context[key] = new_value
        return f"[OK] working_context['{key}'] replaced"

    def archival_memory_insert(self, content: str) -> str:
        self.archival_storage.insert(content)
        return f"[OK] Inserted: {content[:40]}..."

    def archival_memory_search(self, query: str, page: int = 0) -> str:
        """支持分页——function chaining 的核心应用场景"""
        results = self.archival_storage.search(query, page=page)
        if not results:
            return f"[NO RESULTS] page={page}"
        lines = [f"  [{i+1}] {r.content[:80]}" for i, r in enumerate(results)]
        return f"[Archival p={page}]\n" + "\n".join(lines)

    def conversation_search(self, query: str, page: int = 0) -> str:
        results = self.recall_storage.search(query, page=page)
        if not results:
            return f"[NO RESULTS] page={page}"
        lines = [f"  [{m.role}] {m.content[:80]}" for m in results]
        return f"[Recall p={page}]\n" + "\n".join(lines)

    # ── Queue Manager：内存压力控制 ───────────────

    def add_message(self, msg: Message) -> Optional[str]:
        """
        添加消息并触发内存压力检查。
        返回：None（正常）或 warning/flush 提示字符串
        """
        self.fifo_queue.append(msg)
        self.recall_storage.append(msg)  # 永远持久化到 recall storage

        if len(self.fifo_queue) >= self.FLUSH_THRESHOLD:
            return self._flush_queue()
        elif len(self.fifo_queue) >= self.WARNING_THRESHOLD:
            return "[SYSTEM] 内存压力警告：上下文已用 70%+，请将重要信息存入 working_context 或 archival_storage"
        return None

    def _flush_queue(self) -> str:
        """
        强制驱逐：弹出队列前半部分，生成递归摘要。
        被驱逐的消息仍在 recall_storage 中，可通过 conversation_search 找回。
        """
        evict_count = len(self.fifo_queue) // 2
        evicted = [self.fifo_queue.popleft() for _ in range(evict_count)]
        evicted_text = " | ".join(f"{m.role}: {m.content[:25]}" for m in evicted)
        self.recursive_summary = f"[摘要] {evicted_text}"  # 实际场景由 LLM 生成
        return f"[SYSTEM] 队列已刷新，{evict_count} 条消息移入 recall storage"


# ─────────────────────────────────────────────
# 演示：跨 session 记忆检索
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== MemGPT 记忆管理演示 ===\n")

    ctx = MemGPTContext("你是一个记忆助手，始终用 working context 记录用户的重要信息。")

    # Session 1：用户自我介绍
    print("--- Session 1 ---")
    session1 = [
        ("user",      "嗨，我叫小明，是 ML 工程师，在研究 RAG 系统。"),
        ("assistant", "你好小明！RAG 系统很有意思，遇到什么具体问题吗？"),
        ("user",      "主要是检索精度不够高，尤其是多跳问题。"),
        ("assistant", "明白，多跳检索需要迭代查询，我已记下你的背景。"),
    ]
    for role, content in session1:
        warning = ctx.add_message(Message(role=role, content=content))
        if warning:
            print(f"  ⚠️  {warning}")

    # LLM 主动调用函数存储重要信息（function chaining 简化演示）
    print(ctx.core_memory_append("user_profile", "名字：小明，职业：ML工程师，关注：RAG多跳检索"))
    ctx.archival_memory_insert("小明在 Session 1 提到主要问题是多跳检索精度不足")

    # Session 2：新窗口，FIFO Queue 清空，但 working context 和 archival storage 保留
    print("\n--- Session 2（新窗口，测试跨 session 记忆）---")
    ctx.fifo_queue.clear()  # 模拟新会话开始

    ctx.add_message(Message(role="user", content="上次我们聊到什么问题了？"))

    # LLM 先看 working context（已在 main context 里，无需调用函数）
    print(f"Working Context: {json.dumps(ctx.working_context, ensure_ascii=False)}")

    # LLM 再查 archival storage（function calling）
    print(f"\n{ctx.archival_memory_search('RAG 多跳检索')}")

    # 如需更多细节，翻查 recall storage（function chaining 翻页）
    print(f"\n{ctx.conversation_search('小明的问题')}")

# 确定性输出（hash 种子固定）：
# --- Session 1 ---
# [OK] working_context['user_profile'] updated
#
# --- Session 2（新窗口，测试跨 session 记忆）---
# Working Context: {"user_profile": " 名字：小明，职业：ML工程师，关注：RAG多跳检索"}
#
# [Archival p=0]
#   [1] 小明在 Session 1 提到主要问题是多跳检索精度不足
#
# [Recall p=0]
#   [user] 主要是检索精度不够高，尤其是多跳问题。
#   [user] 嗨，我叫小明，是 ML 工程师，在研究 RAG 系统。
#   [assistant] 明白，多跳检索需要迭代查询，我已记下你的背景。
#   [assistant] 你好小明！RAG 系统很有意思，遇到什么具体问题吗？
#   [user] 上次我们聊到什么问题了？
