# Day 1: Introduction to Agents

## 1. 学习目标

今天的重点是理解什么是 AI agent，以及它与普通聊天机器人之间的差异。

## 2. Agent 的基本定义

AI agent 不是单纯会聊天的模型，而是能够：

- 接收输入
- 进行推理和规划
- 决定是否调用工具
- 获取外部信息或执行动作
- 根据结果生成最终输出

也就是说，agent 具备“执行任务”的能力，而不仅仅是“生成文本”。

## 3. Agent 的基本循环

输入 → 推理/规划 → 动作/工具调用 → 输出

### 说明

- Input: 用户提出需求或任务
- Reasoning: 模型判断当前任务、上下文和目标
- Action / Tool Use: 调用函数、查询资料、执行命令或请求外部资源
- Output: 返回结果、建议、执行成果或下一步动作

## 4. chatbot 与 agent 的区别

### Chatbot

- 主要是对话生成
- 通常只基于上下文回答问题
- 不一定会主动调用工具或执行动作
- 更偏向“文本输出器”

### Agent

- 能理解任务目标
- 可以做决策并选择下一步
- 可能调用计算器、搜索器、数据库、API 或脚本
- 更接近“任务执行系统”

## 5. 一个简单的 agent 架构示意

```text
User Request
    ↓
LLM / Model
    ↓
Reasoning & Planning
    ↓
Tool / Action Layer
    ↓
External Data / API / Function
    ↓
Response to User
```

## 6. 关键理解

一个 AI 应用被认为是 agent，不是因为它“看起来像 AI”，而是因为它具备：

- 任务感知能力
- 选择工具或动作的能力
- 能根据环境做调整
- 能将输入转化为实际结果

## 7. 我的学习总结

我认为一个 agent 的核心价值在于：

它不只是“回答问题”，而是“在任务目标下主动完成行动”。

这使得它能够处理更复杂的工作场景，例如：

- 搜索资料
- 分析数据
- 调用 API
- 生成报告
- 执行多步骤任务

## 8. Day 1 任务完成情况

- [ ] 完成 Day 1a
- [ ] 完成 Day 1b
- [ ] 记录 agent loop
- [ ] 记录架构思路
- [ ] 总结 chatbot 与 agent 的区别

## 9. 额外问题记录

- agent 与普通 LLM 应用的边界是什么？
- 什么时候需要工具调用？
- 复杂任务为什么往往需要多步 reasoning？

## 10. 下一步计划

下一步我会继续学习：

- Agent Tools
- MCP
- Session 与 Memory
- Agent Quality 与 Evaluation

到时候会在对应的 notes 文件中继续记录。
