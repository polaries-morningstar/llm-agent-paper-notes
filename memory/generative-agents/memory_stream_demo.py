"""
复现 Generative Agents 的 Memory Stream 与三维检索机制
来源：Park et al. 2023 - Generative Agents: Interactive Simulacra of Human Behavior

依赖：numpy（pip install numpy）
运行：python memory_stream_demo.py
"""

import time
from dataclasses import dataclass
import numpy as np


# ─────────────────────────────────────────────
# 数据结构：单条记忆
# ─────────────────────────────────────────────

@dataclass
class MemoryRecord:
    content: str
    created_at: float       # Unix 时间戳（游戏时间）
    last_accessed: float    # 最后访问时间，影响时近性衰减
    importance: float       # 0.0-1.0，写入时由 LLM 打分后归一化
    embedding: np.ndarray   # 文本嵌入向量，用于计算相关性
    record_type: str = "observation"  # observation / reflection / plan


# ─────────────────────────────────────────────
# Memory Stream：外挂记忆数据库
# ─────────────────────────────────────────────

class MemoryStream:
    """
    论文核心数据结构：外部化的长期记忆库。
    解决 LLM 上下文窗口无法容纳完整经历的根本问题。
    """
    def __init__(self, decay_factor: float = 0.995):
        self.records: list[MemoryRecord] = []
        # 0.995^24 ≈ 0.887，即 24 小时后时近性分数剩 88.7%
        self.decay_factor = decay_factor

    def add(self, content: str, importance_raw: float,
            embedding: np.ndarray, record_type: str = "observation") -> MemoryRecord:
        """写入新记忆。重要性分数在此一次性归一化，不在检索时重算。"""
        now = time.time()
        record = MemoryRecord(
            content=content,
            created_at=now,
            last_accessed=now,
            importance=importance_raw / 10.0,   # 归一化到 [0, 1]
            embedding=embedding,
            record_type=record_type,
        )
        self.records.append(record)
        return record

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> list[MemoryRecord]:
        """
        三维评分检索，等权求和：
          score(m) = recency(m) + importance(m) + relevance(m, query)
        """
        if not self.records:
            return []

        now = time.time()
        scored = []

        for rec in self.records:
            # 时近性：指数衰减，Δt 单位为小时
            hours_elapsed = (now - rec.last_accessed) / 3600
            recency = self.decay_factor ** hours_elapsed      # ∈ (0, 1]

            # 重要性：写入时已归一化
            importance = rec.importance                        # ∈ [0, 1]

            # 相关性：余弦相似度，映射到 [0, 1]
            cos_sim = np.dot(query_embedding, rec.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(rec.embedding) + 1e-8
            )
            relevance = (cos_sim + 1.0) / 2.0                # ∈ [0, 1]

            final_score = recency + importance + relevance    # 最大值 = 3.0
            scored.append((final_score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_records = [rec for _, rec in scored[:top_k]]

        # 更新访问时间（下次检索时 recency 从这里开始衰减）
        for rec in top_records:
            rec.last_accessed = now

        return top_records


# ─────────────────────────────────────────────
# Generative Agent
# ─────────────────────────────────────────────

class GenerativeAgent:
    REFLECTION_THRESHOLD = 150  # importance 累计超过此值时触发反思

    def __init__(self, name: str):
        self.name = name
        self.memory = MemoryStream()
        self._importance_accumulator = 0.0

    def perceive(self, observation: str, importance: float, embedding: np.ndarray):
        """感知事件，写入记忆流，必要时触发反思。"""
        self.memory.add(observation, importance, embedding)
        self._importance_accumulator += importance
        print(f"  [{self.name}] 记录：{observation[:45]}（重要性={importance:.0f}）")

        if self._importance_accumulator >= self.REFLECTION_THRESHOLD:
            self._reflect()
            self._importance_accumulator = 0

    def _reflect(self):
        """
        反思触发（简化版）。
        真实流程：
        1. 取最近 100 条记忆
        2. LLM 生成 3 个最值得关注的高层次问题
        3. 用问题检索记忆流，提炼 5 条洞见（含引用依据）
        4. 洞见写回 Memory Stream（record_type="reflection"）
        """
        print(f"\n  ★ [{self.name}] 触发反思（importance 累计达阈值）")
        # 示意：此处调用 LLM 生成反思内容并写回
        # insight = llm.generate_reflection(self.memory.records[-100:])
        # embed = get_embedding(insight)
        # self.memory.add(insight, importance=8.0, embedding=embed, record_type="reflection")


# ─────────────────────────────────────────────
# 演示：Isabella 的一天
# ─────────────────────────────────────────────

def mock_embed(text: str) -> np.ndarray:
    """固定 hash 种子，输出确定性嵌入向量（实际使用 text-embedding-ada-002 等模型）"""
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(128).astype(np.float32)


if __name__ == "__main__":
    print("=== Memory Stream 演示 ===\n")
    isabella = GenerativeAgent("Isabella Rodriguez")

    # 写入一天的经历
    events = [
        ("Isabella is making coffee at Oak Hill Cafe",         2.0),
        ("A customer complained about the coffee temperature",  4.0),
        ("Klaus Mueller came in and discussed gentrification",  6.0),
        ("Isabella decided to host a Valentine's Day party",   9.0),
        ("Isabella invited Maria Lopez to the party",          4.0),
        ("Maria confirmed she will attend the party",          5.0),
        ("Isabella closed the cafe for the evening",           2.0),
    ]

    print("--- 感知阶段 ---")
    for content, importance in events:
        isabella.perceive(content, importance, mock_embed(content))

    print("\n--- 检索阶段 ---")
    query = "What should Isabella prepare for the Valentine's Day party tonight?"
    results = isabella.memory.retrieve(mock_embed(query), top_k=3)

    print(f"查询：{query}")
    print("Top-3 检索结果：")
    for i, rec in enumerate(results, 1):
        print(f"  {i}. [{rec.record_type}] {rec.content}")

# 确定性输出（hash 种子固定）：
# Top-3 检索结果：
#   1. [observation] Isabella decided to host a Valentine's Day party
#   2. [observation] Maria confirmed she will attend the party
#   3. [observation] Klaus Mueller came in and discussed gentrification
