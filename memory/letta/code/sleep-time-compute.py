"""
模拟 Sleep-time Compute 的核心两阶段机制
来源：Lin et al. 2025 - Sleep-time Compute: Beyond Inference Scaling at Test-time
依赖：无（纯 Python 标准库，不需要 LLM API）
运行：python sleep_time_demo.py
"""

from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# 数据结构：两阶段上下文
# ─────────────────────────────────────────────

@dataclass
class SleepTimeContext:
    """
    封装 sleep-time compute 的状态：
    - raw_context:       原始上下文 c（合同文本/对话历史/代码仓库）
    - processed_context: sleep 后的 c'（LLM 重新思考的产物）
    - rethink_calls:     LLM 调用 rethink_memory 的次数（上限 10）
    """
    raw_context: str
    processed_context: Optional[str] = None
    rethink_calls: int = 0
    MAX_RETHINK: int = 10

    @property
    def is_ready(self) -> bool:
        return self.processed_context is not None


# ─────────────────────────────────────────────
# Sleep-time 函数调用接口（模拟 function calling）
# ─────────────────────────────────────────────

def rethink_memory(ctx: SleepTimeContext, new_content: str) -> str:
    """
    LLM 在 sleep 阶段主动调用的函数。
    用新的推理内容替换当前的 processed context。
    类比：LLM 在草稿纸上不断修改自己的"预备笔记"。
    """
    if ctx.rethink_calls >= ctx.MAX_RETHINK:
        return "[ERROR] 已达到最大 rethink 次数"
    ctx.processed_context = new_content
    ctx.rethink_calls += 1
    return f"[OK] Memory updated（第 {ctx.rethink_calls}/{ctx.MAX_RETHINK} 次 rethink）"


def finish_rethinking(ctx: SleepTimeContext) -> str:
    """LLM 在 sleep 阶段完成后调用，告知系统预处理完毕"""
    if ctx.processed_context is None:
        return "[ERROR] 尚未执行任何 rethink，请先调用 rethink_memory"
    return f"[OK] Sleep-time 完成，共执行 {ctx.rethink_calls} 次 rethink"


# ─────────────────────────────────────────────
# Sleep-time 阶段：LLM 提前"预思考"上下文
# ─────────────────────────────────────────────

def mock_sleep_time_llm(ctx: SleepTimeContext) -> None:
    """
    模拟 LLM 在 sleep 阶段的行为：
    分析原始上下文，推断可能的查询方向，预计算有用结论。

    实际场景：调用支持 function calling 的 LLM，
    提示词："重新组织和整合记忆，生成新见解、新推论和新假设。"
    """
    print(f"\n[Sleep-time] 正在分析上下文（{len(ctx.raw_context)} 字符）...")

    # 第一轮：提取关键数字和实体，发现换算关系
    first_pass = (
        "## 关键信息提取（第一轮 rethink）\n"
        "- 果园面积：10 平方码\n"
        "- 每 2/3 平方码有 87 颗葡萄（需换算为每平方码）\n"
        "  换算：每平方码 = 87 ÷ (2/3) = 87 × 1.5 = 130.5 颗\n"
        "- 收获周期：每 12 个月一次\n"
        "- 每年总收成 = 10 × 130.5 = 1305 颗\n"
        "推断：用户最可能询问 N 年内的总收成"
    )
    print(f"  {rethink_memory(ctx, first_pass)}")

    # 第二轮：基于第一轮结论，预计算常见查询场景
    second_pass = (
        "## 预计算结论（第二轮 rethink）\n"
        "每年收成 = 10 × (87 ÷ (2/3)) = 1305 颗（已验证换算）\n\n"
        "按年预计算：\n"
        "- 1 年：1305 颗\n"
        "- 2 年：2610 颗  ← 最可能被问到\n"
        "- 3 年：3915 颗\n"
        "- 5 年：6525 颗\n\n"
        "注意：若问题涉及半年等非整年周期，按 1305/2 = 652.5 颗折算。"
    )
    print(f"  {rethink_memory(ctx, second_pass)}")
    print(f"  {finish_rethinking(ctx)}")


# ─────────────────────────────────────────────
# Test-time 阶段：使用预处理上下文回答查询
# ─────────────────────────────────────────────

def mock_test_time_llm(ctx: SleepTimeContext, query: str) -> str:
    """
    使用 processed_context（c'）而非原始 raw_context（c）回答查询。
    优势：c' 已经包含预计算结论，LLM 不需要从头推理。
    """
    if not ctx.is_ready:
        context_to_use = ctx.raw_context
        mode = "test-only（无 sleep 预处理）"
    else:
        context_to_use = ctx.processed_context
        mode = "sleep+test（使用预处理上下文）"

    print(f"\n[Test-time] 模式：{mode}")
    print(f"  查询：{query}")
    print(f"  上下文长度：{len(context_to_use)} 字符")

    # 模拟 LLM 在预处理上下文里查找/计算答案
    if ctx.is_ready and "2 年" in query and "2610" in (ctx.processed_context or ""):
        answer = "2610 颗（预处理阶段已计算，直接引用）"
        token_usage = "~15 tokens（仅需定位结论）"
    elif ctx.is_ready and "3 年" in query and "3915" in (ctx.processed_context or ""):
        answer = "3915 颗（预处理阶段已计算，直接引用）"
        token_usage = "~15 tokens（仅需定位结论）"
    elif ctx.is_ready and "5 年" in query and "6525" in (ctx.processed_context or ""):
        answer = "6525 颗（预处理阶段已计算，直接引用）"
        token_usage = "~15 tokens（仅需定位结论）"
    else:
        answer = "需从头推理：10 × (87÷(2/3)) × N = 1305N 颗"
        token_usage = "~200 tokens（需完整换算和推理链）"

    return f"  回答：{answer}\n  Token 使用：{token_usage}"


# ─────────────────────────────────────────────
# 演示：有/无 sleep-time 的对比 + 多查询摊销
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Sleep-time Compute 两阶段演示 ===")

    # 原始上下文 c：数学题背景设置（不含具体问题）
    raw_context = (
        "优素福有 10 平方码的葡萄田。"
        "每 2/3 平方码有 87 颗葡萄。"
        "优素福每 12 个月收获一次葡萄。"
    )

    # ── 方案 A：test-only（无 sleep）──
    print("\n--- 方案 A：Test-only（无 sleep-time）---")
    ctx_no_sleep = SleepTimeContext(raw_context=raw_context)
    print(mock_test_time_llm(ctx_no_sleep, "优素福在 2 年内能收获多少颗葡萄？"))

    # ── 方案 B：sleep + test ──
    print("\n--- 方案 B：Sleep-time + Test-time ---")
    ctx_with_sleep = SleepTimeContext(raw_context=raw_context)
    mock_sleep_time_llm(ctx_with_sleep)  # 空闲时间预处理
    print(mock_test_time_llm(ctx_with_sleep, "优素福在 2 年内能收获多少颗葡萄？"))

    # ── 多查询摊销：一次 sleep，多次 test（体现经济优势）──
    print("\n--- 多查询摊销：一次 sleep，三次 test ---")
    for q in [
        "3 年内能收获多少颗？",
        "5 年内能收获多少颗？",
    ]:
        print(mock_test_time_llm(ctx_with_sleep, q))
    print("\n[摊销] 1 次 sleep 开销 / 3 次查询 = 每次平均成本显著下降")
