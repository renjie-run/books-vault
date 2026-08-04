# AI Agent 开发：从入门到精通

> 整理自 linux.do 社区精华讨论 + 业界最佳实践，按照循序渐进的方式构建完整知识体系。

---

## 📖 阅读指引

| 阶段 | 目标读者 | 核心收获 |
|------|---------|---------|
| 一、入门篇 | 零基础 / 想了解 Agent 是什么 | 理解概念、能跑通第一个 Agent |
| 二、基础篇 | 有 Python 基础 / 想动手 | 掌握核心组件，能搭简单 Agent |
| 三、进阶篇 | 已会用框架 / 想深入 | 了解主流框架差异，做出实用 Agent |
| 四、高级篇 | 有项目经验 / 想精进 | 多 Agent 协作、RAG、记忆系统 |
| 五、精通篇 | 准备上线 / 团队 TL | 评估体系、生产部署、安全与监控 |
| 六、框架全景图 | 所有人 | 一张图看清 AI Agent 全貌 |

---

# 一、入门篇：什么是 AI Agent

## 1.1 一句话定义

> **AI Agent（智能体）= LLM（大脑）+ 工具（手脚）+ 规划（思维链）+ 记忆（经验）**

它不是一个只会聊天的 Bot，而是一个**能自主感知环境、制定计划、调用工具、完成复杂任务的智能程序**。

## 1.2 Agent vs 传统 Chatbot

| 维度 | 传统 Chatbot | AI Agent |
|------|-------------|----------|
| 交互方式 | 一问一答 | 多步自主执行 |
| 能力边界 | 仅生成文本 | 调用 API、操作文件、搜索网页 |
| 记忆 | 当前对话 | 短期+长期记忆 |
| 任务复杂度 | 简单 QA | 多步骤、有依赖关系的复杂任务 |
| 典型例子 | ChatGPT 基础对话 | AutoGPT、Cursor、Devin |

## 1.3 核心公式

$$ \text{Agent} = \text{LLM} + \text{Tools} + \text{Planning} + \text{Memory} $$

- **LLM**：大脑 —— 推理、理解、生成
- **Tools**：手脚 —— 搜索、代码执行、API 调用、数据库查询
- **Planning**：思维 —— 任务分解（Task Decomposition）、自我反思（Self-Reflection）
- **Memory**：经验 —— 短期记忆（上下文窗口）、长期记忆（向量数据库）

## 1.4 五分钟跑通你的第一个 Agent

使用 **OpenAI SDK / LangChain** 最简示例：

```python
# pip install openai
from openai import OpenAI

client = OpenAI()

# 定义一个"工具"——Agent 可以调用的函数
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}：晴，25°C"

# 让 LLM 自主决定是否调用工具
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    }]
)

# Agent 自主决定调用 get_weather("北京")
print(response.choices[0].message.tool_calls)
```

> ✅ 恭喜！你已经理解了 Agent 最核心的循环：**感知 → 推理 → 行动 → 观察 → 再推理...**

---

# 二、基础篇：核心组件深度解析

## 2.1 LLM —— 选择合适的大脑

| 模型 | 优势 | 适用场景 |
|------|------|---------|
| GPT-4o / GPT-4 Turbo | 综合最强，function calling 稳定 | 通用 Agent、复杂推理 |
| Claude 3.5 Sonnet | 代码能力强，长上下文 | 代码 Agent、设计 Agent |
| Gemini 1.5 Pro | 100 万 token 上下文 | 超长文档分析 |
| DeepSeek-V3 | 性价比极高，中文优秀 | 中文场景、成本敏感 |
| Qwen-Max | 中文理解顶级 | 国内业务 |
| Llama 3.1 405B | 开源最强，可私有部署 | 数据安全场景 |
| 本地小模型 (7B-70B) | 低延迟、免费 | 简单任务路由、分类 |

### 选模型原则

1. **原型验证**：先用最强模型（GPT-4o / Claude 3.5），确认思路可行
2. **成本优化**：用强模型做"规划"，用小模型做"执行"
3. **数据安全**：敏感数据用开源模型本地部署

## 2.2 Tools —— 赋予 Agent 行动力

### 工具类型

```
┌─────────────────────────────────────────────┐
│                  Agent Tools                 │
├───────────┬───────────┬──────────┬─────────┤
│  知识类   │  操作类   │  感知类  │ 协作类  │
├───────────┼───────────┼──────────┼─────────┤
│ Web 搜索  │ 代码执行  │ 图片识别 │ MCP     │
│ 知识库RAG │ API 调用  │ 语音识别 │ 多Agent│
│ 文档检索  │ 数据库CRUD│ 视频理解 │ 人机协作│
│ 维基查询  │ 文件操作  │ 网页抓取 │         │
└───────────┴───────────┴──────────┴─────────┘
```

### Function Calling 最佳实践

```python
# ✅ 好的工具定义：描述清晰、参数明确、有示例
{
    "name": "search_knowledge_base",
    "description": "在企业知识库中搜索相关内容。适用于查询公司制度、技术文档、历史决策等。",
    "parameters": {
        "query": "搜索关键词，建议使用完整的问题或短语",
        "top_k": "返回结果数量，默认 5，最大 20",
        "filter": "可选，按文档类型过滤：policy/technical/meeting"
    }
}

# ❌ 差的工具定义
{
    "name": "search",
    "description": "搜索",
    "parameters": {"q": "查询"}
}
```

> 🔗 关联阅读：你 vault 中的 [[../01-Porjects/AI/02.prompt]] 提到「Prompt = 设计文档 + 我们要让他做的事情」，工具定义本质上也是一种 Prompt 工程。

## 2.3 Planning —— 让 Agent 会思考

### ReAct 模式（Reasoning + Acting）

最经典的 Agent 推理框架：

```
Thought: 我需要知道北京今天的天气才能回答用户
Action: get_weather("北京")
Observation: 北京：晴，25°C
Thought: 我已经拿到天气信息，可以回答用户了
Answer: 北京今天晴天，气温 25°C，适合出行！
```

### 其他规划策略

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| **ReAct** | 思考→行动→观察，循环迭代 | 通用任务 |
| **Chain-of-Thought** | 逐步推理，不调用工具 | 数学、逻辑推理 |
| **Tree-of-Thought** | 多分支探索，回溯选择 | 复杂决策 |
| **Plan-and-Execute** | 先制定完整计划，再逐步执行 | 多步骤确定性任务 |
| **Reflexion** | 执行后自我反思，改进下一次 | 需要迭代优化的任务 |

## 2.4 Memory —— 让 Agent 有记性

```
┌──────────────────────────────────────────┐
│              Agent Memory                │
├──────────────────┬───────────────────────┤
│   短期记忆        │      长期记忆         │
│ (上下文窗口)      │   (向量数据库)         │
├──────────────────┼───────────────────────┤
│ • 当前对话历史    │ • 用户偏好持久化       │
│ • 本次任务中间结果 │ • 历史对话摘要         │
│ • 工具调用记录    │ • 知识库检索结果       │
│ 容量：~128K token │ 容量：近乎无限          │
└──────────────────┴───────────────────────┘
```

### 记忆管理策略

1. **滑动窗口**：只保留最近 N 轮对话
2. **摘要压缩**：用 LLM 将历史对话压缩成摘要
3. **向量检索**：将历史存入向量数据库，按相关性检索
4. **混合策略**：摘要 + 向量检索，兼顾效率和精度

---

# 三、进阶篇：主流框架与实战

## 3.1 框架选型指南

| 框架 | 定位 | 适合人群 | 优点 | 缺点 |
|------|------|---------|------|------|
| **LangChain** | 全能型 LLM 应用框架 | 初学者、快速原型 | 生态丰富、文档多 | 封装过重、调试难 |
| **LangGraph** | 有状态多步骤 Agent | 复杂工作流 | 图结构清晰、可控 | 学习曲线陡 |
| **CrewAI** | 多 Agent 协作 | 多角色协作场景 | 角色定义直观 | 灵活性有限 |
| **AutoGen** (微软) | 多 Agent 对话 | 企业级、微软生态 | 对话驱动、灵活 | 社区相对小 |
| **Dify / Coze** | 低代码 Agent 平台 | 非开发者、快速验证 | 可视化、易上手 | 定制性受限 |
| **OpenAI Agents SDK** | 原生 Agent | 深度 OpenAI 用户 | 简洁、高性能 | 绑定 OpenAI |
| **Vercel AI SDK** | 前端友好的 AI SDK | 全栈开发者 | 流式一流 | 后端能力弱 |
| **Semantic Kernel** | 企业级 AI 编排 | .NET/Java 企业 | 微软官方支持 | 生态不如 Python |

### 选框架的思考框架

```
你的需求是什么？
    │
    ├─ 只是简单对话 + 工具调用 → OpenAI SDK / Vercel AI SDK
    ├─ 需要复杂工作流编排 → LangGraph
    ├─ 需要多 Agent 协作 → CrewAI / AutoGen
    ├─ 团队没有工程师 → Dify / Coze
    └─ 企业级 .NET 栈 → Semantic Kernel
```

## 3.2 LangChain / LangGraph 实战

### LangChain 核心概念

```python
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 1. 定义工具
@tool
def calculator(expression: str) -> str:
    """执行数学计算，输入数学表达式"""
    return str(eval(expression))

# 2. 创建 Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_openai_functions_agent(llm, [calculator])

# 3. 执行
result = agent.invoke({"input": "(123 + 456) * 789 等于多少？"})
```

### LangGraph —— 状态图驱动

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    next_step: str

# 定义节点（每个节点是一个处理步骤）
def router(state: AgentState):
    """路由：决定下一步做什么"""
    # 判断是否需要工具调用 → tool_executor / END

def tool_executor(state: AgentState):
    """执行工具调用"""

# 构建图
graph = StateGraph(AgentState)
graph.add_node("router", router)
graph.add_node("tools", tool_executor)
graph.add_conditional_edges("router", decide_next, {
    "tools": "tools",
    "end": END
})
graph.add_edge("tools", "router")  # 工具执行后回到路由
graph.set_entry_point("router")

app = graph.compile()
```

> 🔑 **LangGraph 的核心价值**：把 Agent 循环建模为**有向图**，每个节点的输入输出是确定的 State，这使得调试、测试、可视化都变得清晰。

## 3.3 多 Agent 协作（CrewAI）

```python
from crewai import Agent, Task, Crew

# 定义角色
researcher = Agent(
    role="研究员",
    goal="深度调研指定主题，输出结构化的调研报告",
    backstory="你是一位经验丰富的研究分析师",
    llm="gpt-4"
)

writer = Agent(
    role="技术写手",
    goal="将调研报告转化为通俗易懂的技术文章",
    backstory="你擅长将复杂概念讲得清晰易懂",
    llm="gpt-4"
)

# 定义任务
research_task = Task(description="调研 AI Agent 的最新进展", agent=researcher)
writing_task = Task(description="基于调研报告撰写一篇科普文章", agent=writer)

# 组建 Crew
crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
result = crew.kickoff()
```

### 多 Agent 设计的黄金法则

1. **角色单一职责**：一个 Agent 只做一件事，做好一件事
2. **信息单向流动**：避免 Agent 之间的循环依赖
3. **人类在回路中**：关键决策节点插入人工审核
4. **不是越多越好**：2-3 个 Agent 解决大部分问题

---

# 四、高级篇：RAG、记忆与多模态

## 4.1 RAG（检索增强生成）

```
用户问题
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Embedding  │ ──▶ │  向量数据库   │ ──▶ │  Top-K 文档  │
│  向量化查询  │     │  (相似度搜索) │     │  (相关上下文) │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
                    ┌─────────────────────────────────────┐
                    │  LLM 生成                             │
                    │  "根据以下上下文回答用户问题..."        │
                    └─────────────────────────────────────┘
```

### RAG 进阶技巧

| 技巧 | 说明 | 效果 |
|------|------|------|
| **HyDE** | 先让 LLM 生成假设答案，再用答案检索 | 提升召回率 |
| **Re-ranking** | 初检索后用 Cross-Encoder 重排序 | 提升精度 |
| **Small-to-Big** | 检索句子，返回段落 | 平衡精度和上下文 |
| **Self-RAG** | LLM 自我判断是否需要检索 | 减少无效检索 |
| **Graph RAG** | 用知识图谱代替向量检索 | 处理实体关系 |
| **Agentic RAG** | Agent 自主决定检索策略 | 自适应、最灵活 |

## 4.2 记忆系统架构

```python
# 分层记忆架构示例
class AgentMemory:
    def __init__(self):
        self.sensory_memory = []      # 当前交互的原始消息
        self.short_term = []          # 最近 N 轮对话（上下文窗口）
        self.long_term = VectorDB()   # 持久化语义记忆
        self.episodic = []            # 关键事件记忆

    def remember(self, query):
        """混合检索：短期 + 长期 + 情景"""
        short = self.short_term[-10:]                          # 最近 10 轮
        long = self.long_term.search(query, top_k=5)           # 语义检索
        episode = self.retrieve_episodes(query)                 # 情景匹配
        return self.merge(short, long, episode)

    def consolidate(self):
        """记忆巩固：定期将短期记忆压缩写入长期"""
        summary = llm.summarize(self.short_term)
        self.long_term.store(summary)
        self.short_term = self.short_term[-5:]  # 保留最近 5 轮
```

## 4.3 多模态 Agent

现代 Agent 越来越不局限于文本：

| 模态 | 工具/模型 | 能力 |
|------|----------|------|
| 图像理解 | GPT-4V, Claude 3.5 Vision | 看图分析、UI 识别 |
| 图像生成 | DALL-E 3, Stable Diffusion | 根据描述生成图片 |
| 代码执行 | Python REPL, E2B Sandbox | 执行代码并返回结果 |
| 浏览器操作 | Playwright, Puppeteer | 自动化网页操作 |
| 语音 | Whisper, TTS | 语音输入输出 |

---

# 五、精通篇：生产级 Agent

## 5.1 评估体系

### 一个成熟的 Agent 评估应从三个维度切入

```
           ┌──────────────┐
           │   评估维度    │
           └──────┬───────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌───────┐   ┌─────────┐   ┌─────────┐
│ 功能   │   │  质量    │   │  安全    │
│ 评估   │   │  评估    │   │  评估    │
├───────┤   ├─────────┤   ├─────────┤
│任务完成率│  │ 输出准确性│   │ 注入测试 │
│工具调用率│  │ 回答相关性│   │ 权限检查 │
│Step效率 │  │ 用户满意度│   │ 数据泄露 │
└───────┘   └─────────┘   └─────────┘
```

### 自动化评测方法

```python
# 1. LLM-as-Judge：用强模型评价弱模型
eval_result = judge_llm.evaluate(
    question="...",
    agent_answer="...",
    expected_answer="...",
    criteria=["准确性", "完整性", "简洁性"]
)

# 2. 工具调用正确率
assert tool_called == "get_weather"
assert tool_args == {"city": "北京"}

# 3. 轨迹评估（Trajectory Eval）
# 判断 Agent 的中间步骤是否合理
# 是否走了不必要的弯路？是否遗漏了关键步骤？
```

## 5.2 生产部署架构

```
                          ┌──────────────┐
                          │   负载均衡     │
                          └──────┬───────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │  Agent 实例 1 │  │  Agent 实例 2 │  │  Agent 实例 N │
      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │   向量数据库   │  │    缓存       │  │   消息队列    │
      │ (Milvus/PG)  │  │  (Redis)     │  │  (Kafka)    │
      └──────────────┘  └──────────────┘  └──────────────┘
```

### 关键决策

| 问题 | 推荐方案 |
|------|---------|
| LLM 调用不可靠？ | 重试机制 + 降级策略 + 多模型 Fallback |
| 推理太慢？ | 流式输出 + 异步执行 + 小模型预处理 |
| Token 成本高？ | 缓存常见问题 + Prompt 压缩 + 模型蒸馏 |
| 幻觉怎么办？ | RAG 约束 + 事实性校验 + 人工兜底 |
| 上下文溢出？ | 滑动窗口 + 摘要 + 关键信息提取 |

## 5.3 安全防护清单

```
□ Prompt Injection 防护：输入清洗、指令隔离
□ 工具调用权限控制：最小权限原则
□ 敏感信息过滤：PII 检测 + 脱敏
□ 输出审核：内容安全过滤器
□ 速率限制：防止滥用
□ 审计日志：完整记录 Agent 决策轨迹
□ 人类兜底：高风险操作必须人工确认
```

## 5.4 可观测性

```python
# 使用 LangSmith / LangFuse / Phoenix 进行追踪
# 核心监控指标：
metrics = {
    "latency_p50": 1200,      # ms, 中位延迟
    "latency_p99": 5000,      # ms, 99分位延迟
    "token_usage": 15000,     # 每次对话 token 消耗
    "tool_call_rate": 0.7,    # 工具调用比例
    "success_rate": 0.92,     # 任务成功率
    "hallucination_rate": 0.03,  # 幻觉率
    "user_satisfaction": 4.2, # 用户评分 (1-5)
}
```

---

# 六、AI Agent 开发框架全景图

```
                              AI Agent 开发全景

    ┌─────────────────────────────────────────────────────────────┐
    │                        应用层                                │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
    │  │ 代码 Agent│  │ 数据 Agent│  │ 客服 Agent│  │  自动化 Agent │ │
    │  │ Cursor   │  │ 数据分析  │  │ 智能客服  │  │  RPA + AI    │ │
    │  │ Devin    │  │ 报表生成  │  │ 知识问答  │  │  工作流自动化 │ │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
    └─────────────────────────────────────────────────────────────┘
                                  │
    ┌─────────────────────────────────────────────────────────────┐
    │                        框架层                                │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
    │  │LangChain │  │ LangGraph│  │  CrewAI  │  │   AutoGen    │ │
    │  │ 通用框架  │  │ 状态图编排│  │ 多Agent  │  │  对话式Agent │ │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
    │  │  Dify    │  │   Coze   │  │OpenAI SDK│  │ Vercel AI    │ │
    │  │ 低代码平台│  │  Bot平台  │  │ 原生 SDK │  │  全栈 SDK    │ │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
    └─────────────────────────────────────────────────────────────┘
                                  │
    ┌─────────────────────────────────────────────────────────────┐
    │                        能力层                                │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
    │  │   LLM    │  │   RAG    │  │  工具调用 │  │   记忆系统    │ │
    │  │ GPT-4o   │  │ 向量检索  │  │  Fn Call │  │  短期+长期   │ │
    │  │ Claude   │  │ 知识图谱  │  │  MCP协议 │  │  向量数据库  │ │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
    └─────────────────────────────────────────────────────────────┘
                                  │
    ┌─────────────────────────────────────────────────────────────┐
    │                        设施层                                │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
    │  │ 可观测性  │  │  评估    │  │  安全     │  │   部署       │ │
    │  │ LangFuse │  │ LLM Judge│  │ Guardrails│  │  Docker/K8s │ │
    │  │ Phoenix  │  │ 自动化评测│  │ 权限控制  │  │   Serverless│ │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
    └─────────────────────────────────────────────────────────────┘
```

---

## Excalidraw 框架图建议

你可以在 Excalidraw 中按以下结构绘制一张**AI Agent 学习路线图**：

```
主题：AI Agent 开发从入门到精通

中心圆 → "AI Agent"
    │
    ├─ 左上: 🟢 入门篇
    │   ├─ Agent 概念 & 公式
    │   ├─ vs 传统 Chatbot
    │   └─ 第一个 Agent Demo
    │
    ├─ 右上: 🟡 基础篇
    │   ├─ LLM 选型
    │   ├─ Tools 设计
    │   ├─ ReAct 规划
    │   └─ Memory 架构
    │
    ├─ 左下: 🟠 进阶篇
    │   ├─ 框架选型 (LangChain/LangGraph/CrewAI)
    │   ├─ 多 Agent 协作
    │   └─ 实战项目
    │
    ├─ 右下: 🔴 高级篇
    │   ├─ RAG 进阶 (HyDE/Graph RAG)
    │   ├─ 记忆系统
    │   └─ 多模态 Agent
    │
    └─ 底部: 🟣 精通篇
        ├─ 评估体系 (LLM-as-Judge)
        ├─ 生产部署 & 监控
        ├─ 安全防护
        └─ 成本优化
```

**绘图步骤：**
1. 在 Obsidian 中 `Ctrl+P` → 输入 `Excalidraw: Create new drawing`
2. 从中心开始画圆 → "AI Agent"
3. 向五个方向延伸，分别画阶段框
4. 用箭头连接，标注依赖关系
5. 用不同颜色区分阶段（绿→黄→橙→红→紫）
6. 加入关键图标/emoji 增强可读性

---

## 📚 推荐学习资源

| 类型 | 资源 | 说明 |
|------|------|------|
| 📖 论文 | *ReAct: Synergizing Reasoning and Acting in Language Models* | Agent 推理范式的奠基论文 |
| 📖 论文 | *AutoGPT* / *BabyAGI* | 自主 Agent 的早期经典 |
| 📖 论文 | *Generative Agents: Interactive Simulacra* | 斯坦福小镇，记忆系统经典 |
| 🛠️ 框架 | [LangChain](https://www.langchain.com/) | 最全面的 LLM 应用框架 |
| 🛠️ 框架 | [LangGraph](https://langchain-ai.github.io/langgraph/) | 有状态 Agent 编排 |
| 🛠️ 框架 | [CrewAI](https://www.crewai.com/) | 多 Agent 协作框架 |
| 🛠️ 平台 | [Dify](https://dify.ai/) | 低代码 Agent 构建平台 |
| 🎓 课程 | Andrew Ng《AI Agentic Design Patterns》 | 吴恩达的 Agent 设计模式 |
| 🎓 课程 | 《Building Systems with ChatGPT API》 | DeepLearning.AI 实战课 |
| 📝 博客 | [Lilian Weng 的 Agent 博客](https://lilianweng.github.io/) | OpenAI 研究员的深度分享 |

---

> 💡 **下一步**：建议从 1.4 节的 Demo 开始动手，然后根据你的实际需求选择一个框架深入。LLM 时代的学习关键是 **"先跑起来，边做边学"**。
>
> 🔗 关联你 Vault 中的笔记：[[../01-Porjects/AI/01.start|名词解释]] | [[../01-Porjects/AI/02.prompt|Prompt 工程]] | [[../01-Porjects/AI/03.guidelines|Vibe Coding 准则]] | [[../01-Porjects/AI/04.OpenDesign|Open Design]]

---

# 七、大厂 JD 拆解篇：AI Agent 研发工程师到底要会什么？

> 📌 来源：小红书 [知乎大模型实战] 整理的多家大厂 AI Agent 研发岗 JD 综合分析

## 7.1 两类 AI Agent 工程师岗位

大厂 AI Agent 研发岗并非只有一种，而是明显分为**两个层次**：

| 维度 | 🔹 基础工程岗 | 🔹 中高级自研岗 |
|------|-------------|----------------|
| **薪资区间** | 15-30K | 35-60K |
| **核心能力** | 框架二次开发 + 工程落地 | 底层架构自研 + 算法优化 |
| **关键技术** | RAG 搭建、多智能体编排、分布式链路、安全观测体系 | 分层记忆架构（从零搭建）、底层检索优化、混合检索调度 |
| **框架要求** | 熟练使用 LangChain / LangGraph / Dify 等主流框架 | 能突破框架限制，自研底层组件 |
| **典型产出** | 基于现有框架搭建可用的 Agent 系统 | 从零设计 Agent 基础设施，产出可复用的底层能力 |
| **面试门槛** | 能跑通完整项目流程 | 能展示底层理解 + 项目作品集 |
| **一句话总结** | **会用**工具搭 Agent | **能造** Agent 基础设施 |

## 7.2 自学的四大痛点 & 如何突破

根据多份 JD 的交叉分析，自学容易踩的坑和对应的破解法：

| 痛点 | 表现 | 破解方法 |
|------|------|---------|
| **知识点零散** | 学了 RAG、Agent、MCP 但串不起来 | 按「应用层 → 框架层 → 能力层 → 设施层」全链路打通 |
| **无完整落地流程** | 只会跑 Demo，不懂从需求到上线 | 跟一个完整的企业级项目：需求分析 → 架构设计 → 开发 → 测试 → 部署 → 监控 |
| **Demo 达不到招聘标准** | 面试时只能讲玩具项目 | 对标 JD 定制项目：记忆系统 + 混合检索 + 多 Agent + 安全观测 |
| **拿不出底层能力证明** | 面试一问底层就露馅（"纯调包选手"） | 深入理解记忆调度原理、向量检索底层实现、LLM 推理优化 |

## 7.3 大厂 JD 高频关键词 → 能力映射

从多家大厂 JD 中提取的高频要求，一一对应到具体能力：

### 🔑 高频关键词一览

```
RAG · 多智能体 · 分布式 · 安全观测 · 分层记忆 · 混合检索
LangChain · MCP · 向量数据库 · Prompt 工程 · 长文本 · 幻觉控制
```

### 📋 能力清单逐项拆解

| JD 关键词 | 面试会怎么问 | 你需要掌握到什么程度 |
|-----------|-------------|---------------------|
| **RAG 搭建** | 如何设计一个企业级 RAG 系统？ | 不止会调 `vectorstore.similarity_search()`，要懂分块策略、Embedding 选型、Re-ranking、HyDE 等进阶技巧 |
| **多智能体** | 多个 Agent 如何协作？状态怎么管理？ | 能设计角色分工、信息流、冲突解决机制；熟悉 CrewAI / AutoGen 的角色定义模式 |
| **分布式链路** | Agent 调用链如何追踪？如何定位瓶颈？ | 会用 LangSmith / LangFuse / Phoenix 做全链路追踪，能从 trace 中分析性能瓶颈 |
| **安全观测** | 如何防止 Prompt Injection？如何审计 Agent 行为？ | 理解输入输出过滤、权限控制、审计日志、人类兜底机制 |
| **分层记忆** | 短期记忆和长期记忆如何协同？ | 能设计滑动窗口 + 摘要压缩 + 向量检索的混合记忆架构 |
| **混合检索调优** | 关键词匹配 + 语义搜索如何融合？ | 理解 BM25 + Dense Retrieval 的融合策略、RRF (Reciprocal Rank Fusion) 等算法 |
| **长文本处理** | 上下文超长怎么办？ | 掌握滑动窗口、摘要压缩、关键信息提取、分块总结等策略 |
| **幻觉控制** | 如何降低 Agent 的幻觉率？ | RAG 约束 + 事实性校验 + 引用溯源 + 不确定性声明 |

## 7.4 学习路线图：从能投简历到能拿 Offer

```
阶段一：基础工程入门（目标：能投基础工程岗）
├─ 掌握 LangChain / LangGraph 核心用法
├─ 完成一个完整的 RAG + Agent 项目（从文档到可用系统）
├─ 理解 MCP 协议，能接入外部工具
├─ 了解 Prompt Engineering 最佳实践
└─ 产出：一个可演示的企业级 Agent 应用

阶段二：底层能力突破（目标：能冲中高级岗位）
├─ 深入理解向量检索原理（HNSW、IVF、量化等）
├─ 从零实现分层记忆架构（不依赖框架封装）
├─ 掌握混合检索调优（BM25 + Dense + Re-rank）
├─ 设计多 Agent 协作系统（含状态管理、冲突解决）
└─ 产出：一个自研底层组件的 Agent 项目 + 技术博客

阶段三：生产级能力（目标：拿到高薪 Offer）
├─ 全链路可观测性（trace → span → metrics）
├─ 安全防护体系（注入防御、权限控制、内容审核）
├─ 分布式部署与性能优化
├─ 评估体系建设（自动化评测 + 人工评估）
└─ 产出：一个经得起面试深挖的生产级作品集
```

## 7.5 面试作品集 Checklist

面试时面试官最想看到的东西，按优先级排列：

- [ ] **完整项目链路**：不是孤立 Demo，而是从需求 → 设计 → 实现 → 部署的全流程
- [ ] **底层理解展示**：能解释"为什么选这个分块大小""为什么用 HNSW 而不是 IVF"
- [ ] **问题解决记录**：开发过程中踩过什么坑？怎么解决的？（侧面证明你没走捷径）
- [ ] **性能数据**：你的 RAG 召回率是多少？延迟是多少？跟基准对比怎么样？
- [ ] **架构图**：能用一张图把系统架构讲清楚
- [ ] **代码仓库**：GitHub 上有可审查的项目代码，不是 Jupyter Notebook 碎片
- [ ] **技术博客**：1-2 篇深度技术文章，展示你的思考深度

---

> ⚡ **核心结论**：大厂 AI Agent 岗的核心分水岭在于 —— **你是"框架使用者"还是"底层构建者"**。从框架使用者跨越到底层构建者，关键是补齐**记忆架构、检索优化、安全观测**这三块硬骨头。这与你之前整理的 [[../01-Porjects/AI/02.prompt|Prompt 工程]] 和 [[../01-Porjects/AI/03.guidelines|Vibe Coding 准则]] 相辅相成 —— Prompt 是"术"，底层能力是"道"，二者缺一不可。
>
> 🔗 关联你 Vault 中的笔记：[[../01-Porjects/AI/01.start|名词解释]] | [[../01-Porjects/AI/02.prompt|Prompt 工程]] | [[../01-Porjects/AI/03.guidelines|Vibe Coding 准则]] | [[../01-Porjects/AI/04.OpenDesign|Open Design]]
