# LLM & Agent 论文精读笔记

> 系统整理 LLM / Agent 领域的论文精读、架构分析与框架解析，持续更新。

---

## 目录结构

```
llm-agent-paper-notes/
├── agent-framework/     # Agent 架构与框架论文
├── memory/              # Memory 机制相关论文
├── frameworks/          # 主流框架源码与设计解析
│   ├── pydantic-ai/     # PydanticAI 框架
│   ├── langchain/       # LangChain 框架
│   └── llamaindex/      # LlamaIndex 框架
├── rag/                 # RAG 检索增强生成
├── reasoning/           # 推理与规划（CoT、ReAct 等）
├── multimodal/          # 多模态 Agent
└── resources/           # 参考资料、工具与学习路径
```

---

## 分类说明

### Agent 架构与框架 (`agent-framework/`)
围绕 Agent 的设计范式展开，包括：
- 经典架构论文：ReAct、AutoGPT、BabyAGI、AgentBench 等
- 多 Agent 协作：MetaGPT、AutoGen、CrewAI 等
- Agent 评测与 Benchmark

### Memory 机制 (`memory/`)
Agent 的记忆系统设计，包括：
- 短期 / 长期记忆架构
- Memory 检索与更新策略
- 代表论文：MemGPT、A-MEM、HippoRAG 等

### 主流框架解析 (`frameworks/`)
对开源框架的设计理念、核心抽象与源码进行拆解：
- **PydanticAI**：类型安全的 Agent 构建框架
- **LangChain / LangGraph**：工具链与图执行引擎
- **LlamaIndex**：以数据为中心的 RAG 框架

### RAG 检索增强生成 (`rag/`)
- Naive RAG / Advanced RAG / Modular RAG
- 向量检索、重排序、GraphRAG
- 代表论文：Self-RAG、RAPTOR、GraphRAG 等

### 推理与规划 (`reasoning/`)
- Chain-of-Thought、Tree-of-Thought、Graph-of-Thought
- ReAct、Reflexion、LATS
- Planning 与 Tool Use

### 多模态 Agent (`multimodal/`)
- 视觉-语言 Agent
- GUI Agent（Web / Desktop 操控）
- 代表工作：AppAgent、WebVoyager 等

### 资源汇总 (`resources/`)
- 推荐学习路径
- 常用工具与数据集
- 相关博客、课程与综述

---

## 笔记格式

每篇论文笔记建议包含以下结构：

```markdown
# 论文标题

- **来源**：arXiv / NeurIPS / ICML 等
- **时间**：YYYY-MM
- **链接**：

## 核心问题
> 这篇论文解决了什么问题？

## 方法概述

## 关键模块 / 架构图

## 实验结论

## 个人思考

## 参考
```

---

## 贡献

欢迎提 Issue 或 PR，一起完善笔记内容。
