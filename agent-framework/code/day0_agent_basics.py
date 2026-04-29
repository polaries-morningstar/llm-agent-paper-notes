"""
Day 0 博客配套代码：AI Agent 核心概念演示
来源：HuggingFace Agents Course Unit 1-4
"""

import inspect
import random


# ─────────────────────────────────────────────
# Part 1: @tool 装饰器——自动生成 LLM 可读的工具描述
# ─────────────────────────────────────────────

def tool(func):
    """将 Python 函数包装成 Agent 可用的工具"""
    signature = inspect.signature(func)
    # 处理无类型标注参数，fallback 为 "Any"
    arguments = [
        (p.name, p.annotation.__name__
         if p.annotation != inspect.Parameter.empty else "Any")
        for p in signature.parameters.values()
    ]
    return_ann = signature.return_annotation
    outputs = (return_ann.__name__
               if return_ann != inspect.Parameter.empty else "Any")

    func.to_string = lambda: (
        f"Tool Name: {func.__name__}, "
        f"Description: {func.__doc__}, "
        f"Arguments: {', '.join(f'{n}: {t}' for n, t in arguments)}, "
        f"Outputs: {outputs}"
    )
    return func


@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气信息，包含温度、湿度和天气状况"""
    mock_data = {
        "北京": "晴，25°C，湿度 40%",
        "New York": "多云，15°C，湿度 60%",
        "上海": "阴，20°C，湿度 75%",
    }
    # 模拟 20% 的概率网络超时（演示失败路径）
    if random.random() < 0.2:
        raise ConnectionError("timeout after 5s")
    return mock_data.get(city, f"{city}: 数据暂不可用")


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """向指定邮件地址发送一封邮件。适用于需要通知他人或分发报告的场景。"""
    print(f"[Mock] 邮件已发送 → {recipient} | 主题: {subject}")
    return f"邮件已成功发送给 {recipient}"


# ─────────────────────────────────────────────
# 演示 1：工具描述 vs 工具执行（两个不同角色）
# ─────────────────────────────────────────────

print("=" * 60)
print("演示 1：LLM 看到的工具描述（注入 System Prompt 的内容）")
print("=" * 60)
print(get_weather.to_string())
# 输出：
# Tool Name: get_weather, Description: 获取指定城市的当前天气信息，包含温度、湿度和天气状况,
# Arguments: city: str, Outputs: str

print()
print(send_email.to_string())
# 输出：
# Tool Name: send_email, Description: 向指定邮件地址发送一封邮件。适用于需要通知他人或分发报告的场景.,
# Arguments: recipient: str, subject: str, body: str, Outputs: str

print()
print("框架实际调用工具时的执行结果（Observation）")
# 临时关闭随机失败以保证演示稳定
random.seed(42)
result = get_weather("北京")
print(result)
# 输出：晴，25°C，湿度 40%


# ─────────────────────────────────────────────
# 演示 2：工具描述质量对比（批判性分析配套）
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("演示 2：低质量 vs 高质量工具描述对比")
print("=" * 60)


@tool
def fetch(q: str) -> str:
    """fetch data"""
    return "mock data"


@tool
def search_recent_news(query: str) -> str:
    """搜索过去 24 小时内与 query 相关的新闻标题和摘要。
    适用于：需要实时信息、训练数据截止日期之后发生的事件。
    不适用于：历史事件、数学计算、代码生成。
    返回：最多 5 条新闻，每条包含标题 + 来源 + 一句话摘要。"""
    return "[1] 标题: OpenAI 发布 GPT-5 | 来源: TechCrunch | 摘要: OpenAI 今日宣布..."


print("低质量描述（LLM 无法判断何时调用）：")
print(fetch.to_string())
# Tool Name: fetch, Description: fetch data, Arguments: q: str, Outputs: str

print()
print("高质量描述（触发条件、适用场景、返回格式全部说明）：")
print(search_recent_news.to_string())
# Tool Name: search_recent_news, Description: 搜索过去 24 小时内与 query 相关的新闻标题和摘要。...

print()
print("实际调用结果对比：")
print(f"fetch('OpenAI news')        → {fetch('OpenAI news')}")
print(f"search_recent_news('OpenAI') → {search_recent_news('OpenAI')}")


# ─────────────────────────────────────────────
# 演示 3：Think-Act-Observe 循环（含成功路径和失败路径）
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("演示 3：Think-Act-Observe 循环（成功路径 + 失败路径）")
print("=" * 60)

TOOLS = {"get_weather": get_weather, "send_email": send_email}


def simulate_agent(user_query: str, max_steps: int = 5, seed: int = None):
    """
    模拟 Agent 的 Think-Act-Observe 循环。
    真实框架中，Think 步骤由 LLM 完成；此处用规则模拟以便演示。
    """
    if seed is not None:
        random.seed(seed)

    print(f"\n{'='*60}")
    print(f"用户: {user_query}")
    print(f"{'='*60}")

    context = [{"role": "user", "content": user_query}]
    retry_count = 0

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} ---")

        # ── Think（真实框架中由 LLM 完成）──
        last_content = context[-1]["content"] if context else ""
        if "ERROR" in last_content and retry_count < 2:
            # 失败路径：LLM 读到错误 Observation 后决定重试
            thought = f"上一次调用失败：{last_content}。重试一次。"
            action = {"tool": "get_weather", "args": {"city": "New York"}}
            retry_count += 1
        elif any(c["role"] == "tool" and "ERROR" not in c["content"] for c in context):
            # 成功路径：已获得有效工具结果，准备生成最终回复
            obs = next(c["content"] for c in reversed(context) if c["role"] == "tool")
            thought = f"已获得天气数据（{obs}），任务达成，可以回复用户。"
            action = None
        else:
            # 初始状态：决定调用天气工具
            thought = "用户要实时天气，训练数据没有今天的信息，应该调用 get_weather 工具。"
            action = {"tool": "get_weather", "args": {"city": "New York"}}

        print(f"[Thought]  {thought}")

        if action is None:
            obs = next(c["content"] for c in reversed(context) if c["role"] == "tool")
            final_answer = f"纽约目前天气：{obs}，希望对您有帮助。"
            print(f"[Final]    {final_answer}")
            break

        # ── Act（框架解析 LLM 输出并调用工具）──
        tool_func = TOOLS.get(action["tool"])
        print(f"[Act]      调用 {action['tool']}({action['args']})")

        try:
            # ── Observe（成功路径）──
            observation = tool_func(**action["args"])
            print(f"[Observe]  ✓ {observation}")
            context.append({"role": "tool", "content": observation})
        except Exception as e:
            # ── Observe（失败路径）──
            error_msg = f"ERROR: {e}"
            print(f"[Observe]  ✗ 工具调用失败: {e}，将重试或换策略")
            context.append({"role": "tool", "content": error_msg})


# 运行演示（seed=42 时 random.random() > 0.2，走成功路径；seed=0 时可能触发失败）
simulate_agent("纽约现在天气怎样？", seed=42)   # 大概率成功路径
simulate_agent("纽约现在天气怎样？", seed=0)    # 可能触发失败路径演示重试


# ─────────────────────────────────────────────
# 演示 4：Chat Template 序列化（需要 transformers）
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("演示 4：Chat Template 序列化（需安装 transformers）")
print("=" * 60)

try:
    from transformers import AutoTokenizer

    messages = [
        {"role": "system",    "content": "你是一个专业的客服助手。"},
        {"role": "user",      "content": "我的订单 ORDER-123 出了问题"},
        {"role": "assistant", "content": "您好，请问是什么问题？"},
        {"role": "user",      "content": "一直显示配送中，三天了"},
    ]

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(prompt)
    # 输出（SmolLM2 格式）：
    # <|im_start|>system
    # 你是一个专业的客服助手。<|im_end|>
    # <|im_start|>user
    # 我的订单 ORDER-123 出了问题<|im_end|>
    # <|im_start|>assistant
    # 您好，请问是什么问题？<|im_end|>
    # <|im_start|>user
    # 一直显示配送中，三天了<|im_end|>
    # <|im_start|>assistant

except ImportError:
    print("请先安装 transformers：pip install transformers")
