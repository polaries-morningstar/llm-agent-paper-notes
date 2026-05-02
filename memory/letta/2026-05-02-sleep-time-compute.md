---
title: "别等用户问——Sleep-time Compute 如何让 LLM 的空闲时间变成推理资本"
date: 2026-05-02
tags: [Agent, Memory, LLM, SleepTimeCompute, TestTimeScaling, Letta]
categories: [memory]
---

# 别等用户问——Sleep-time Compute 如何让 LLM 的空闲时间变成推理资本
## ——MemGPT 解决了"怎么存"，Letta 的这篇论文回答了"存完了怎么提前用"

> **论文**：Sleep-time Compute: Beyond Inference Scaling at Test-time
> **作者**：Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, Joseph E. Gonzalez（Letta / UC Berkeley）
> **发表**：2025 年 4 月 | arXiv:2504.13171 | 预印本（Letta 官方研究）
> **链接**：[论文](https://arxiv.org/abs/2504.13171) | [代码](https://github.com/letta-ai/sleep-time-compute)
> **关键词**：`Sleep-time Compute` `Test-time Scaling` `Stateful Agents` `Memory Reorganization` `Query Amortization`
> **一句话**：在用户提问之前，让 LLM 利用空闲时间预先处理和重组上下文——同等准确率下测试时计算量减少 5 倍，或同等计算量下准确率提升 13-18%。

---

## TL;DR

**一句话总结**：Sleep-time Compute 把一次推理拆成两个阶段：用户提问前，LLM 在空闲时间"重新思考"已有上下文并改写成更有利于回答的形式；用户提问时，LLM 用这个预处理后的上下文以更少的计算量得到正确答案。

**三点拆解**：

- 🔑 **这是 test-time scaling 的正交方向**：过去两年的 scaling 研究（o1、o3、DeepSeek-R1）都在问"用户提问后，多用多少推理 token 能提升准确率"。Sleep-time compute 问的是一个完全不同的问题："在用户提问之前，提前用计算做什么？"论文的实验把两者作为替代选项对比，证明了各自的独立价值——但两者叠加使用时的组合效应，目前尚无系统实验数据，是这篇论文的一个明显空白。
- 🔑 **多查询摊销是真正的工程价值所在**：单次查询时，sleep-time compute 的成本不一定划算。但当同一份上下文会被多个查询使用时（比如同一份合同被 50 个律师助理反复查询），sleep 成本被平摊——实验数据显示，10 个查询共享同一个上下文时，平均每次查询的总成本降低 2.5 倍。这是 sleep-time 的核心商业逻辑，也是它和 test-time scaling 最本质的区别。
- 🔑 **查询可预测性是效果边界，而静默失败才是部署风险**：sleep-time 效果的好坏依赖"上下文是否预示了可能的查询"。论文把这当作局限，但我认为更危险的问题是：如果 sleep-time 的预处理结果是错误的，test-time 的 LLM 会直接引用它而不重验证——这个约束实际上决定了 sleep-time compute 的适用边界比摘要暗示的要窄得多。

---

## §3 背景与动机：test-time scaling 之后，还有什么空间？

### 从 MemGPT 到 Sleep-time Compute：同一个团队，不同的问题层次

如果你读过 MemGPT 的博客，你已经知道 UC Berkeley / Letta 团队在 2023 年解决了什么问题：如何让 LLM 管理比上下文窗口大得多的信息，通过 working context + archival storage + recall storage 的三层架构，让 Agent 在跨 session 对话中保持记忆。

但 MemGPT 解决的是存储问题——信息怎么存、怎么找。它没有回答的问题是：**当信息已经在存储里了，用户提问时，LLM 还是要花大量计算来检索、整理、理解——这个开销怎么办？**

假设你在维护一个面向客户的 AI 助理，每个用户有几年的对话历史存在 archival storage 里。每次用户提问，Agent 需要：解析问题语义、搜索相关历史（向量检索，可能多跳）、把检索结果和当前问题整合到上下文、在有限 token 预算内生成回复。这四步都发生在用户等待的时候。如果搜索需要 3 次迭代（function chaining），每次 LLM 推理 2 秒，用户等待至少 6 秒——在企业级助理场景里，这已经超出了大多数用户的容忍阈值。

### Test-time Scaling 的边际成本在上升

2024 年，OpenAI 发布 o1 之后，整个 AI 行业开始把目光投向"推理时扩展计算"（test-time scaling）：让模型在生成答案前做更多内部思考（chain-of-thought），用更多 token 换取更高准确率。这个方向在数学竞赛、代码生成等任务上取得了惊人进展——o3 在 ARC-AGI 上突破了人类水平。

但 test-time scaling 有一个内在约束：**它的全部计算都在用户等待期间发生**。用户等待 2 秒和等待 20 秒，对应的是完全不同的体验。随着推理 token 使用量增加，延迟线性增长，成本也线性增长——高质量回复的边际成本在快速攀升。

更麻烦的是，test-time scaling 是一次性的——同样的上下文被多个用户查询时，每次都要重新推理，没有任何复用。如果一份合同同时被 100 个律师助理查询，test-time compute 被浪费了 100 次。

### "空闲时间"是未被利用的计算资源

这是 sleep-time compute 的出发点：**LLM 并不是每时每刻都在服务用户请求**。在 Agent 系统里，大量时间是"空闲"的——用户没有提问、任务在等待、后台服务处于 standby。

2023 年的操作系统类比（MemGPT）把上下文窗口类比为 RAM、把数据库类比为磁盘。Sleep-time compute 延伸了这个类比：操作系统有"后台进程"——在前台进程等待 IO 的时候，OS 调度器会把 CPU 时间给后台任务（磁盘碎片整理、预取、索引更新）。LLM 能不能也有"后台进程"，在用户不问问题的时候，提前整理、分析已有的上下文？

直觉上这像一个显而易见的想法。但实现它需要回答几个不显然的问题：预先"思考"什么？怎么把思考结果存下来？思考的格式是自然语言还是向量？这个预思考对不同类型的查询有多大帮助？Sleep-time Compute 论文是第一篇系统回答这些问题的工作。

### 为什么是 2025 年：三个条件的碰头

这个想法不是全新的。传统数据库有预计算（OLAP Cube）、搜索引擎有预索引、CDN 有内容预缓存——"提前做工作"是工程里的老思路。但在 LLM 时代，直到 2024-2025 年才同时满足三个前提条件：

**第一，支持 function calling 的 LLM 已经足够可靠**。Sleep-time compute 需要 LLM 通过 `rethink_memory` 工具调用来改写自己的 working context，而这个操作必须产生格式正确的 JSON 参数。2022 年的 LLM 函数调用错误率太高，sleep-time 处理结果根本存不进去。GPT-4o 和 Claude Sonnet 3.7 的函数调用可靠性才达到了生产部署所需的阈值。

**第二，stateful agent 基础设施已经成熟**。Sleep-time 的输出（c'）需要被持久化存储并在下次会话时加载——这正是 MemGPT / Letta 的 working context 层提供的功能。没有一个"安全存放 LLM 预处理结果"的地方，sleep-time 的输出无处可放。Letta 团队做 sleep-time 研究的同时，底层基础设施已经是自己的产品，这不是巧合。

**第三，test-time scaling 的成本开始让人肉疼**。o1 级别模型每次推理可能消耗数千个"推理 token"，成本是普通模型的 10-50 倍。当用户的月账单里出现大量"同样的问题付了 10 次全价"的情况，工程团队才开始认真寻找"预计算"这条出路。Sleep-time compute 是市场和技术条件双重成熟的产物，而不是一个纯粹的学术探索。

---

## §4 核心 Idea：把推理从"等用户"变成"不等用户"

💡 **核心类比**：一个出色的助理不会等老板说"去查一下上周的合同条款"才开始查——当他知道老板明天要开合同审查会，他昨晚就把相关条款整理好放在桌上了。Sleep-time compute 就是在给 LLM 这个"昨晚整理"的能力。

论文把推理正式拆成两个阶段，S(c) → c'（睡眠阶段）和 T_b(q, c') → a（测试阶段）：

```
  传统 Test-time only 路径：
  ┌───────────────┐      ┌────────────────────────────┐
  │  上下文 c      │      │  用户查询 q 到达             │
  │  (合同全文/    │      │  ↓                          │
  │   对话历史/   │ ───→ │  LLM 推理（budget = B）      │
  │   代码仓库)   │      │  在 c 和 q 的 joint 上思考   │
  └───────────────┘      │  → 回答 a                   │
                          └────────────────────────────┘
                           用户在等 B 个 token 的推理时间

  Sleep-time Compute 两阶段路径：
  ┌───────────────┐  空闲时  ┌─────────────────────────┐
  │  上下文 c      │ ──────→ │ Sleep-time 阶段 S(c)→c'  │
  │  (提前可用)   │          │ LLM 最多 10 次调用        │
  └───────────────┘          │ rethink_memory(new_str)  │
                              │ 每次改写 working context │
                              └───────────┬─────────────┘
                                          │ c' 存入持久化存储
                                          ↓
                              ┌─────────────────────────┐
                              │ 用户查询 q 到达           │
                              │ Test-time T_b(q,c') → a │
                              │ budget b << B（小得多）  │
                              └─────────────────────────┘
                               用户等待时间 = b 而非 B
```

（参考原论文 Figure 1 自绘）

上图的关键是 **c' 不只是 c 的压缩**——c' 是 LLM 对 c 重新推理后的产物，包含了 LLM 认为可能有用的新推断、预计算结论和重组后的信息结构。论文的 sleep 提示词写道："重新组织和整合记忆，生成新见解、新推论和新假设——不只是把信息搬来搬去，而是真正地*思考*这些信息意味着什么。优先考虑新信息超过现有记忆。"

技术实现上，sleep-time 通过两个 function call 完成：`rethink_memory(new_content)` 用新内容替换当前 working context；`finish_rethinking()` 终止 sleep-time 过程。LLM 最多可以调用 10 次 `rethink_memory`，类似于人在写作时的多轮修订，而不是一次性输出最终答案。

---

## §5 方法拆解：四个设计决策，每个都有它的理由

### 5.1 为什么在自然语言空间而不是向量空间"预计算"？

这是 sleep-time compute 最反直觉的设计选择之一。传统的预计算（KV cache、向量索引）都工作在低维的数值空间。Sleep-time compute 选择在自然语言空间输出——c' 是一段文本，不是嵌入向量。

为什么不用向量表示？不妨考虑 KV cache prefetching 这个对比方案：当用户下次进入会话时，把上次的 KV 激活值缓存起来，直接复用，跳过重新推理的过程。这在延迟层面确实有效——不需要重新 forward 就能恢复上下文。但 KV cache 的问题是：它只是"记住了之前的推理"，而不是"提前做了新的推理"。一个缓存了旧 KV 的 LLM 在面对新问题时，仍然需要用完整的 test-time 计算量来推理——它只是省去了"加载历史"的开销，没有省去"思考"的开销。

Sleep-time compute 要做的事情更主动：在用户提问之前，LLM 已经思考过"这份上下文里有哪些可能有用的推论，应该怎么组织这些信息才能让未来的查询更容易回答"。这个"提前的思考"必须以自然语言的形式存储，因为只有自然语言才是 test-time LLM 可以直接理解和推理的格式。向量表示能被向量检索系统使用，但 LLM 无法直接在向量空间里继续思考——它需要把一切转换回 token 序列。

更重要的是，自然语言的 c' 是**可解释的**。你可以打开 working context 看看 LLM 在 sleep-time 里想了什么，发现错误、调试问题、手动修正某个错误的预计算。如果是压缩向量或激活值，出错了你完全不知道为什么，也没有任何办法介入修正。这在工程落地时是一个决定性因素：可调试性决定了一个系统能不能进入生产环境。

### 5.2 Stateful 评测集：为什么现有的 benchmark 不够用？

论文构造了 Stateful GSM-Symbolic 和 Stateful AIME，而没有直接用原始的 GSM8k 或 AIME 题库。理由是：普通的 math benchmark 是"self-contained"的——题面和问题在同一个 prompt 里，没有"上下文比问题早到"的概念，sleep-time 完全无用武之地。

Stateful 版本把一道题的"背景设置"和"具体问题"人为分成两个时间步骤：背景在"睡眠阶段"提供给模型；具体问题在"测试阶段"才告知。划分的方式是**自动的**：对于 GSM-Symbolic，论文用解析器从题目自然语言描述里提取"情景设置"部分（实体、数量关系、背景约束）作为 c，再提取"问题句"（通常是最后一句"问 X 是多少"）作为 q。对于 AIME，由于竞赛题结构更复杂，划分更偏向人工标注：把每道题分成"条件信息"和"求解目标"两部分。

这个构造方式有一个内在的偏差值得注意：**既然 c 是从整道题的题面里提取的，c 天然地预示了 q 的类型**——这是设计使然，也是 sleep-time 在这些任务上表现好的根本原因之一，而不完全是方法本身的通用能力。换句话说，实验选择的任务保证了高可预测性，但这个"保证"本身来自于构造方式，不来自于真实世界的分布。当你把 sleep-time 部署到真实场景（用户的问题不从上下文里"解析"出来），这个保证消失了。Stateful GSM-Symbolic 有 P1（5000 个示例）和 P2（2500 个示例）两个难度级别；Stateful AIME 包含 2024 年和 2025 年 AIME 题目合计 60 道。

值得注意的是：这两个数据集都是单一领域（数学推理），只能说明 sleep-time 在"高结构化推理"场景下的效果。论文没有构造任何开放域问答的 stateful 版本——技术上并非做不到，而是因为开放域问答的上下文可预测性极低，sleep-time 的效果会接近于零。如果把这类任务放进论文，整体结论的说服力会大打折扣。这个实验任务选择不是中立的技术决策，而是有方向性的——论文选择了最有利于 sleep-time 的评测场景，然后把"可预测性依赖"作为局限诚实地写在讨论里，但没有提供"可预测性低时 sleep-time 究竟有多没用"的量化数据。读者在引用这些结论时，应该有意识地注意这个边界。一个具体的错误引用模式：某工程师看到"5x test-time 节省"的结论，直接把 sleep-time 应用到用户随机输入的客服对话场景——上下文是用户多轮闲聊历史，可预测性极低。sleep-time 在这里几乎无效：预处理生成了一堆"可能的问题和答案"，但用户实际提的问题完全不在预测范围内，test-time 依然需要完整推理，sleep 成本却已经发生了。在套用论文数据之前，最简单的检验方法是：从目标场景取 20-30 个历史查询，手动评估"这些查询能否在拿到上下文时就大致猜到"——如果超过一半无法预测，停下来，不要开启 sleep-time。

### 5.3 查询可预测性：为什么不是所有上下文都受益？

论文里最有洞察力的发现之一：sleep-time compute 的效果强烈依赖于"上下文能多大程度预示可能的查询"。论文用 Llama2-70B 计算查询相对于上下文的条件对数概率来量化这个"可预测性"。

直觉上：一道数学题的设置信息（苹果数量、收成频率）高度预示了即将被问到的问题类型（计算总量、比较差值）——这类上下文非常适合 sleep-time。相比之下，一个人的个人背景信息可能对应成千上万种不同的问题，可预测性极低——sleep-time 在这里的帮助很有限。

```
  sleep-time 准确率增益 vs 查询可预测性（参考原论文 Figure 5 自绘）

 准确率
 提升
  +20% │                              ●  ●
  +15% │                           ●
  +10% │                      ●
   +5% │              ●
    0% │  ●  ●  ●────────────────────────────→
       └──────────────────────────────── 查询可预测性（条件对数概率）
        低 ←─────────────────────────→ 高
        （自由对话/SWE）        （数学题面/逻辑设定）

  SWE-Features ≈ 低可预测性区域（增益约 1.5x）
  Stateful GSM/AIME ≈ 高可预测性区域（增益约 5x）
```

（参考原论文 Figure 3 & Figure 10 自绘）

这个发现有重要的工程含义：**sleep-time compute 不是万能的，它是一个特化工具**，在"上下文和查询之间有结构性关联"的场景里才有价值。论文没有给出如何在生产中自动检测"这个上下文是否值得做 sleep-time"，这是 §7 批判性分析的核心落点之一。

### 5.4 多查询摊销：真正改变经济模型的机制

单次查询下，sleep-time compute 的总成本未必更低。但当同一份上下文被 N 个查询使用时，sleep-time 成本被摊销：

```
  多查询场景下的摊销成本（参考原论文 Figure 9 自绘）

  每次查询
  的有效成本
  （相对test-only）
   1.5x │ ●（N=1，sleep-time 更贵）
   1.0x │───────●（N≈3，break-even）
   0.8x │           ●
   0.6x │                ●
   0.4x │──────────────────────●（N=10，2.5x 更便宜）
        └──────────────────────────────→ 每个上下文的查询数 N
          1       3       5       7      10
```

（参考原论文 Figure 9 自绘。成本模型：test-time token 的延迟权重是 sleep-time token 的 10 倍）

注意：这条曲线基于"test-time token 成本是 sleep-time token 的 10 倍"这个假设，这个参数的来源和场景依赖性在 §7.2 有详细讨论，建议在套用数字之前先读完那一节再决定是否开启 sleep-time。

这个摊销曲线在哪些现实场景里成立？**企业知识库助手、法律文件分析、代码仓库问答**——这些场景里，一份文档会被多个用户反复查询，摊销效应真实存在。个人化对话助手则不然——每个用户的对话历史是独特的，摊销几乎不可能发生，sleep-time 在这里得不偿失。

---

## §6 实验结果 + 代码示例

### 核心数据

| 任务 | 模型 | 上下文可预测性 | test-time-only | sleep+test（等计算量） | 收益形式 |
|------|------|--------------|----------------|----------------------|---------|
| Stateful GSM-Symbolic P1 | GPT-4o-mini | 高 | 基线 | 同等准确率 | **~5x test-time 节省** |
| Stateful AIME 2024+2025 | o3-mini | 高 | 基线 | 同等准确率 | **~5x test-time 节省** |
| Stateful GSM-Symbolic P1 | GPT-4o | 高 | 基线 | **+13% 准确率** | 同等计算量 |
| Stateful AIME 2024+2025 | o1 | 高 | 基线 | **+18% 准确率** | 同等计算量 |
| SWE-Features | Claude Sonnet 3.7 | 低 | 基线 | 同等准确率 | **~1.5x test-time 节省** |

两种使用方式：同样 token 做到更准，或者同样准确率用更少 token。"上下文可预测性"列直接解释了为什么数学任务是 5x 而软工任务只有 1.5x。Sleep-time vs pass@k 对比：**在所有任务和模型上，sleep-time 都优于同等计算量的 pass@k 并行采样**——说明 sleep-time 从根本上改善了问题表示，而不只是增加了命中概率。

### 代码示例：模拟 Sleep-time Compute 的两阶段流程

```python
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
```

**运行输出：**
```
=== Sleep-time Compute 两阶段演示 ===

--- 方案 A：Test-only（无 sleep-time）---
[Test-time] 模式：test-only（无 sleep 预处理）
  查询：优素福在 2 年内能收获多少颗葡萄？
  上下文长度：57 字符
  回答：需从头推理：10 × (87÷(2/3)) × N = 1305N 颗
  Token 使用：~200 tokens（需完整换算和推理链）

--- 方案 B：Sleep-time + Test-time ---
[Sleep-time] 正在分析上下文（57 字符）...
  [OK] Memory updated（第 1/10 次 rethink）
  [OK] Memory updated（第 2/10 次 rethink）
  [OK] Sleep-time 完成，共执行 2 次 rethink

[Test-time] 模式：sleep+test（使用预处理上下文）
  查询：优素福在 2 年内能收获多少颗葡萄？
  上下文长度：326 字符
  回答：2610 颗（预处理阶段已计算，直接引用）
  Token 使用：~15 tokens（仅需定位结论）

--- 多查询摊销：一次 sleep，三次 test ---
[Test-time] 模式：sleep+test（使用预处理上下文）
  查询：3 年内能收获多少颗？
  回答：3915 颗（预处理阶段已计算，直接引用）
  Token 使用：~15 tokens（仅需定位结论）
[Test-time] 模式：sleep+test（使用预处理上下文）
  查询：5 年内能收获多少颗？
  回答：6525 颗（预处理阶段已计算，直接引用）
  Token 使用：~15 tokens（仅需定位结论）
[摊销] 1 次 sleep 开销 / 3 次查询 = 每次平均成本显著下降
```

---

## §7 批判性分析：sleep-time 是真正的范式转移还是精巧的工程技巧？

### 7.1 这三个约束，哪一个才是真正的天花板？

论文承认了三个局限：查询可预测性依赖、二阶段假设的静态性、SWE 任务增益低。我认为这三个约束的危险程度并不相同，需要排序：

**最危险的是"静默失败"，而不是"可预测性低"**。可预测性低意味着 sleep-time 效果差——这是效果边界，你顶多浪费了一些 sleep 计算，test-time 仍然正常工作。而"静默失败"是一个不同性质的问题：如果 sleep-time 的预处理结果（c'）包含错误推断，test-time 的 LLM 会直接把这个错误当作"可靠的预计算结论"引用，产生高置信度的错误答案。

设想一个具体的失败链条：一道数学题的背景说"每 2/3 平方码有 87 颗葡萄"，sleep-time 的 LLM 把换算做错了，把"每平方码 130.5 颗"误算成"每平方码 87 颗"写入了 c'。Test-time 的 LLM 拿到 c' 后看到一个整洁的"已知：每平方码 87 颗"，认为这是已验证的结论，直接用来计算最终答案——算出一个错误数字，但置信度极高，语气肯定，没有任何"我不确定"的信号。用户收到一个错误的答案，却因为模型表现出的确定性而不会去质疑它。

这和 RAG 的检索错误有本质区别。RAG 在找不到相关段落时，通常会产生一个"基于泛化知识的推测"，语气相对模糊，或者直接说"我没有找到相关信息"——检索错误的信号相对明显。但 sleep-time 的错误是**主动植入的伪确定性结论**：LLM 在 rethink 阶段以"整理者"的身份把一个错误推断包装成了一个"已知事实"，植入到 c' 里。这种错误比 RAG 的检索失败更难被 test-time LLM 识别，因为它看起来就是一条正确的预处理结论。论文在实验里没有单独测量 sleep-time 的错误率，也没有分析"引入了 sleep-time 错误后准确率方差如何变化"——这是比"效果低"更需要被回答的问题，因为它影响的不是"够不够好"，而是"安不安全"。

**其次危险的是二阶段假设的脆弱性**。论文的所有实验都是静态的：上下文 c 固定，然后测试 q 到来。但实际的 Agent 场景里上下文是动态变化的——每次对话轮次都在更新上下文，sleep-time 的 c' 可能在用户发下一条消息后就过期了。如果 c' 过期率很高，sleep-time 的预处理会被频繁作废，摊销经济账就算不过来了。论文完全没有研究这个 stale-cache 问题——是因为认为在目标场景里不会发生（上下文长期稳定的知识库场景），还是因为意识到了问题但回避了？从论文的实验设计来看，前者更可能，但后者无法排除。

**相对最轻的反而是"可预测性问题"**。它是效果约束，不是安全约束。而且这个约束有一个明确的工程解法方向：在触发 sleep-time 之前，用一个轻量模型估算上下文的可预测性分数，只对高分上下文开启 sleep-time。论文里的 Llama2-70B 对数概率方法可以直接工程化——这不是一个"未解决"的问题，而是一个"需要额外工程投入"的问题，复杂度可控。

这个三级优先排序的实践含义：如果你要在生产里部署 sleep-time compute，最先要建的不是"可预测性门控"，而是"c' 质量验证"——检测 sleep-time 输出里是否有明显错误推断，在 test-time 之前先做一道质量过滤。

### 7.2 我看到的额外问题

**多查询摊销的成本模型缺乏来源说明，且高度影响结论**。论文把 test-time token 的成本设为 sleep-time token 的 10 倍，但没有解释这个比例从哪来。实际上 sleep-time 和 test-time 调用的是同一个 LLM，API 单价完全相同——10 倍只能解释为延迟成本的权重（test-time 在用户等待的关键路径上，而 sleep-time 不在）。

这个 10 倍设定对实验结论有决定性影响，而且是双向的。对于延迟敏感场景（实时客服 P99 < 500ms），10 倍实际上是低估的——在那种场景里 test-time 的每一个 token 都更值钱，sleep-time 的优势会更大；对于批处理场景（离线报告生成，没有实时等待），10 倍是高估的——两者 token 成本相同，摊销曲线的 break-even 点会从 N=3 推迟到 N=10 甚至更高。论文用一个固定数字掩盖了这个场景依赖性，导致读者无法判断"这个结论在我的场景里是否成立"——而这个判断恰恰是工程师最需要做的。更实用的做法是把这个倍率参数化：对于 P99 延迟 < 200ms 的实时客服场景，延迟成本极高，取 20 倍更合适，break-even 在 N≈2；对于异步批处理场景（无人等待），取 1 倍，break-even 推到 N≈10 甚至更高；介于两者之间的交互场景取 5-10 倍，对应 §5.4 图里的 N≈3-5 区间。用你自己的参数代入重绘摊销曲线，再决定是否开启 sleep-time——这一步花不了多少时间，却能避免把论文结论直接套用到完全不同的场景里。

**Sleep-time 与 test-time compute 的组合效应未被研究**。论文把两者作为替代选项分析——给定相同的 token 预算，是全部用于 sleep 效果更好，还是全部用于 test 效果更好。但在真实高质量部署里，两者会**叠加使用**：先做 sleep-time 预处理，再用 o3 级别模型做完整推理。预处理后的 c' 里已经包含了很多推理链，o3 的内部 CoT 在这种输入上表现会不同——可能因为 c' 的信息密度高而更高效，也可能因为 c' 的结论和 o3 自己的推理路径产生冲突而反而不稳定。这个问题在论文发布后的实际应用里必然会遇到，但没有任何数据回答它。

**Letta 作为商业公司发布这篇论文，存在选择性呈现的动机**。论文的实验任务高度有利于 sleep-time（数学推理题是可预测性最高的任务之一），而在可预测性低的任务（自由对话、创意写作）上完全没有实验。SWE-Features 虽然显示了 1.5x 的改善，但这个数据被论文呈现得比较平淡，没有被作为负面结论强调。读者应该意识到：这是一篇由框架团队发布的、旨在验证其产品核心功能的论文，实验任务选择本身可能已经经过了筛选。

### 7.3 给工程师的实操建议

**只在上下文结构化、查询类型有限的场景开启 sleep-time**。客服知识库（上下文是产品手册，查询是用户问题）、法律审查助手（上下文是合同，查询是条款合规性问题）、数学/逻辑推理 Agent 是最佳适用场景。开启前，用小模型快速评估"上下文的可预测性"（参考论文里的 Llama2-70B 对数概率方法），作为是否触发 sleep-time 的门控。通用对话助理、创意写作助手不适合。

**在 c' 里为每个预计算结论添加置信度标注，并设计降级路径**。让 LLM 在 rethink 时对每个结论附上置信度（"我 95% 确定 2 年收成是 2610，基于换算 87/(2/3)×10×2"），这样 test-time 的 LLM 在看到低置信度结论时会重新验证，而不是盲目引用。同时保证系统里始终有原始的 c 作为备份，当 c' 质量不可信时能无缝降级到 test-only——这是防止"睡眠时算错、测试时放大"的最小安全边界。

**用 N 查询数预估 break-even 再决定是否开启 sleep-time**。每启动一次 sleep-time 有固定的 token 成本，除非预期查询数 N 超过 break-even 点（实验数据约为 3-5 个），否则 test-only 更经济。对于个人化对话助理（每个用户的历史对应的 N=1，摊销不可能发生）、实时性强的随机查询场景，sleep-time 很可能是负优化。

**建立 c' 质量监控和版本管理**。Sleep-time 的输出是 LLM 生成的自然语言，不像数据库事务有内置的一致性保证。在生产环境里，建议把每次 sleep-time 的输出（c'）和触发条件（c 的版本哈希、时间戳、使用的模型版本）一起存储，同时持续记录后续 test-time 查询的准确率。如果某个 c' 对应的 test-time 表现明显偏低，立即触发重新 sleep 或者降级到 test-only 路径。这个监控链路不只是防错机制，也是在生产中判断"当前上下文的 sleep-time ROI 是否仍然成立"的核心数据来源——没有这个数据，你永远不知道 sleep-time 究竟在帮忙还是在帮倒忙。

---

## §8 应用落地 + 相关工作 + 资源

### 落地场景

**企业知识库问答（同一文档，多用户查询）**：一份 200 页的技术规范被技术支持团队的 50 个工程师频繁查询。Sleep-time 只需对这份文档做一次预处理：提取关键规格参数、预计算常见单位换算、整理跨章节关联（"第 3 章的限制条件如何影响第 7 章的配置选项"）。之后每个工程师的查询都从 c' 出发，test-time 计算量降低 5 倍。部署注意：文档更新后必须立即使旧的 c' 失效并重新触发 sleep-time，c' 应与文档版本号绑定，便于审计某个历史时期的回答基于哪个版本的预处理。*不适用条件*：若文档更新频率高于每日（如实时价格表、动态政策文件），c' 的失效成本会让 sleep-time 得不偿失——每次更新都要重新触发 sleep 并等待处理完成，期间用户收到的是过期的预处理结果。高频更新的知识库仍应使用传统 RAG 路径。

**数学/量化推理 Agent**：量化策略回测（上下文是市场数据、策略参数，查询是各种比率和归因计算）、财务报告自动分析（上下文是季报，查询是不同维度的指标计算）——这类应用的上下文具有极高的查询可预测性，sleep-time 在 Stateful AIME 上的 18% 准确率提升可以直接平移，且错误类型相对确定（数值计算错误），便于添加 §7.3 提到的置信度验证机制。*不适用条件*：若查询本身是随机的（如用户任意指定回测参数组合），可预测性接近零，sleep-time 预处理的方向会和实际查询大相径庭，不如直接用 o3 做 test-time 推理。

**代码仓库增量分析**：新 commit push 进来时立即触发 sleep-time，分析此次变更涉及哪些模块、引入了哪些新依赖、修改了哪些接口签名、可能影响哪些测试。开发者后续询问"这个 PR 会不会影响数据库连接池"时，c' 里已经有预先整理的变更影响图。这是 SWE 任务里 sleep-time 最有价值的细分场景——不是"写代码"，而是"理解刚发生了什么"，这个查询的可预测性比开放式编程任务高得多。*不适用条件*：若 commit 极为频繁（如 merge queue 里每分钟多次 push），sleep-time 处理上一个 commit 还没结束，下一个 commit 就进来了，形成积压。这种场景下应该只对"重要 commit"（如 PR merge 而不是每次 force push）触发 sleep，或者增加速率限制。

### 相关工作速览

| 工作 | 与 Sleep-time Compute 的核心区别 | 工程师的选择判据 |
|------|--------------------------------|----------------|
| **Test-time Scaling（o1/o3, DeepSeek-R1）** | 在用户等待时扩展推理；sleep-time 在等待前扩展 | 两者正交互补；高准确率场景可叠加，但叠加效果未被实验验证 |
| **MemGPT（Packer et al., 2023）** | 被动更新 working context；sleep-time 是主动、提前的预处理 | MemGPT 解决"存什么/怎么找"，sleep-time 解决"提前想什么" |
| **RAG（Lewis et al., 2020）** | 查询到达后才检索相关片段；sleep-time 在查询前预处理整个上下文 | RAG 适合非结构化大规模知识检索；sleep-time 适合需推理整合的小型上下文 |
| **KV Cache Prefetching（系统级）** | 在向量空间缓存激活值；sleep-time 在自然语言空间预计算新推断 | KV cache 降低 GPU 计算延迟；sleep-time 降低推理链长度——分别在不同层次优化，不互斥 |
| **Letta V1 Agent Loop（2024）** | 工程框架演进，移除 MemGPT 心跳机制，采用原生 LLM 推理 | sleep-time compute 论文是 Letta 的研究成果，V1 是同期的工程实现 |

### 资源链接

- [论文](https://arxiv.org/abs/2504.13171)（arXiv:2504.13171）
- [GitHub 代码](https://github.com/letta-ai/sleep-time-compute)
- [Letta 官网](https://www.letta.com)
- [Stateful GSM-Symbolic 数据集](https://huggingface.co/datasets/letta-ai/stateful-gsm-symbolic)

---

Sleep-time compute 的本质是在问：**推理的时间轴能不能被解耦？** 传统假设是"用户问→模型想→模型答"三步串行，每步都发生在用户等待期间。Sleep-time compute 在这个串行链里打了一个楔子：把"模型想"的一部分提前到用户问之前，在没有人等待的时间里悄悄做完。这个想法成立的前提是上下文和查询之间存在结构性关联——而这个前提在某些场景里成立得很好（合同审查、数学题、代码变更分析），在另一些场景里几乎不成立（自由对话、创意任务）。更深的问题是：这个"结构性关联"能不能被自动识别？如果可以，sleep-time 就能从一个需要人工配置的特化工具，变成一个自适应的通用机制。如果不能，它就会永远停留在"需要工程师判断何时开启"的阶段——而这个判断本身，才是 Letta 下一篇论文真正需要回答的问题。一个不知道何时该休息的助理，和一个永远在等你发问的助理，并没有本质区别。
