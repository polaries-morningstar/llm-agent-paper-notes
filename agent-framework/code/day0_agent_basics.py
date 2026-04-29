"""
Day 0 博客配套代码：AI Agent 核心概念演示
来源：HuggingFace Agents Course Unit 1-4
"""

import inspect


# ─────────────────────────────────────────────
# Part 1: @tool 装饰器——自动生成 LLM 可读的工具描述
# ─────────────────────────────────────────────

def tool(func):
    """将 Python 函数包装成 Agent 可用的工具"""
    signature = inspect.signature(func)
    # 处理无类型标注参数的情况，fallback 为 "Any"
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
    """获取指定城市的当前天气信息"""
    # 此处为演示用 mock 实现，生产中替换为真实天气 API 调用
    mock_data = {
        "北京": "晴，25°C，湿度 40%",
        "New York": "多云，15°C，湿度 60%",
        "上海": "阴，20°C，湿度 75%",
    }
    return mock_data.get(city, f"{city}: 数据暂不可用")


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """向指定收件人发送邮件"""
    # mock 实现
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
# Tool Name: get_weather, Description: 获取指定城市的当前天气信息,
# Arguments: city: str, Outputs: str

print()
print("框架实际调用工具时的执行结果（Observation）")
result = get_weather("北京")
print(result)
# 输出：
# 北京: 晴，25°C，湿度 40%


# ─────────────────────────────────────────────
# Part 2: Think-Act-Observe 循环模拟
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("演示 2：Think-Act-Observe 循环（含成功路径和失败路径）")
print("=" * 60)

TOOLS = {"get_weather": get_weather, "send_email": send_email}


def simulate_agent(user_query: str, max_steps: int = 5):
    """
    模拟 Agent 的 Think-Act-Observe 循环。
    真实框架中，Think 步骤由 LLM 完成；此处用规则模拟以便演示。
    """
    print(f"\n用户: {user_query}\n")
    context = [{"role": "user", "content": user_query}]

    for step in range(1, max_steps + 1):
        print(f"--- Step {step} ---")

        # ── Think（此处用规则模拟 LLM 决策）──
        if step == 1:
            thought = "用户要实时天气，静态知识不够用，应该调用 get_weather 工具。"
            action = {"tool": "get_weather", "args": {"city": "New York"}}
        else:
            thought = "已获得天气数据，任务达成，可以回复用户。"
            action = None

        print(f"[Thought] {thought}")

        if action is None:
            final_answer = f"纽约目前 {context[-1]['content']}，希望对您有帮助。"
            print(f"[Final]   {final_answer}")
            break

        # ── Act（框架调用工具）──
        tool_func = TOOLS.get(action["tool"])
        print(f"[Act]     调用 {action['tool']}({action['args']})")

        try:
            observation = tool_func(**action["args"])
            print(f"[Observe] {observation}")
            context.append({"role": "tool", "content": observation})
        except Exception as e:
            # 失败路径：工具调用出错，循环继续
            print(f"[Observe] 工具调用失败: {e}，将重试或换策略")
            context.append({"role": "tool", "content": f"ERROR: {e}"})


simulate_agent("纽约现在天气怎样？")


# ─────────────────────────────────────────────
# Part 3: Chat Template 演示（需要 transformers）
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("演示 3：Chat Template 序列化（需安装 transformers）")
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
