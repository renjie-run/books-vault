---
tags:
  - deepseek
  - harness
  - agent
  - 架构
source: https://github.com/ht426/deepseek-harness-tutorial/blob/main/01-overview.md
date: 2026-08-17
---

# DeepSeek Harness 架构总结

> 对官方教程第 1 章「总体架构」的白话拆解。原文见 [[DeepSeek harness]]。

## 一句话核心

**dsh = 一个「智能体（agent）组装框架」**，它把下面这些零件拼成一个能跑起来的 agent：

- 调用大模型（LLM）
- 调用工具（tool）
- 记录会话（session log）
- 权限控制、沙箱、人类介入

唯一设计原则：**一切皆插件（everything is a plugin）**。

---

## 三个关键概念

### 1. 一切皆插件 —— 没有「神圣的核心代码」

dsh 跑在一个叫 **Cordis** 的小运行时上，像一个「插线板 / 服务总线」。每个插件往共享的 **Context（上下文）** 里塞三样东西：

- **服务（service）**：提供能力
- **类型化事件（typed event）**：扩展点
- **可逆 effect（effect）**：注册动作本身，卸载时自动撤销

**推论**：模型适配器、工具注册表、会话日志、甚至 agent 循环本身——**全都是插件**，都可以通过配置替换。扩展 = 在别的插件旁再挂一个自己的插件。

> 类比：dsh 不是焊死的汽车，而是一套乐高/宜家家具，每个部件都能拆换。

### 2. Profile 和 Bundle —— 运行中的 dsh 是一棵「插件树」

- **Bundle（捆绑包）**：一个「配置 + 代码」的分发单位，像一盒乐高。
- **Profile（配置档案）**：一个「命名组合」，声明自己叠了哪些 bundle、装了哪些额外插件，并保存用户自己的补丁 `cordis.patch.yml`，像一套具体乐高方案的清单。

启动时按**固定顺序**叠 layer：

```
profile 里列出的 bundle（按顺序）
  → profile 自己的补丁
    → home 级补丁
      → --patch 命令行覆盖层
```

每层都是「补丁」，按 row id 定位某一行并替换，或插入新行。用 `dsh --profile web --dump-config` 打印实际启动出来的树。

> 类比：像 Photoshop 图层，最上层覆盖下层。

### 3. Capability Seam（能力接缝）—— 核心抽象

一个「**可替换的能力**」由三个角色组成，缺一不可：

| 角色 | 作用 | 例子（shell 能力） |
|---|---|---|
| **Service Definition** | 声明接口（拥有 `ctx.<key>`） | `dsh-shell` |
| **Service Provider** | 实现接口 | `dsh-bash-local`、`dsh-bash-sandbox` |
| **Consumer** | 使用服务的一方（常是给模型用的工具） | `dsh-tool-bash` |

**为什么「换一个 provider 整个产品都跟着变」**：`ctx.fs` 和 `ctx.subprocess` 共享同一个「执行世界」——把 provider 指向远程沙箱，Bash、终端、LSP 一起跟着走，无需为每个 provider 分叉版本。

> 类比：Seam 就是「插座 + 插头」——接口（插座）不变，换 provider（插头），所有 consumer（电器）自动用上新能力。

---

## 事件 = 扩展点

三类事件，一句话记：

- **Session 事件**：持久化事实，重载后仍在（写进日志）。
- **Agent 事件**（`agent/*`）：携带活跃 Agent，观察/拦截「在途工作」。
- **Capability 事件**：把策略/适配器挂到接缝上（`fs/*`、`tools/*`…），不引入 agent 循环。

## Turn / Step 流转

- **step（步骤）** = 一次模型请求 + 它调用的工具。
- **turn（轮次）** = 零个或多个 step，输入进来时打开、没活干时关闭。

关键：`agent/pre-step`、`agent/request`、`llm/stream`、`tools/*` 是 **waterfall**（监听器必须调 `next()` 才往下传）；`agent/turn-stopping` 是 serial 且没有 `next()`。

---

## 贯穿全文的不变式（invariant）

> **模型所见 ⟺ 已入日志**：任何进入模型请求的内容，都必须能从日志重建。

所以「想让模型看到新东西，就必须加一个新的 session 事件」。这条铁律保证 fork、恢复、回放、遥测、持久化都从同一条流派生。

---

## 「新行为该放哪」速查表

| 你想做 | 机制 |
|---|---|
| 加模型提供方 | 在 `ctx.llm` 注册适配器 |
| 加面向模型的能力 | 在 `ctx.tools` 注册，schema 进 prompt |
| 加 shell 执行 | 注册 `ctx.shell` 后端 |
| 加文件访问/策略 | 注册 `ctx.fs` provider 或监听 `fs/*` |
| 拦截请求/工具/turn | 用 `agent/*` / `tools/*` 事件 |
| 加模型可见上下文 | `agent.inject()` |
| fork 一个会话 | `ctx.sessions.fork(...)` |
| 把注册限定到某 agent | 用该 agent 的 `agent.ctx` |

---

## 整体心智模型（一句话）

> dsh 是靠 Cordis 运行时把「一堆插件」叠成「插件树」的 agent 框架；每个能力是「接口 + 提供方 + 消费者」三件套的接缝；扩展 = 在正确的扩展点上挂插件或换 provider；模型看到的一切都从日志来。

## 下一步

- 第 2 章：环境与运行，看 `--dump-config` 的插件树
- 第 3 章：Cordis 机制细节
- 第 6–8 章：逐个讲每个接缝
