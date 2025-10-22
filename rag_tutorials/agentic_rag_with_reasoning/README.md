# 🧐 具备推理能力的智能体驱动型检索增强生成（Agentic RAG）

一款功能先进的检索增强生成（RAG）系统，基于 Agno 框架、Claude 模型与 OpenAI 技术，可展示 AI 智能体的**分步推理过程**。该实现支持用户上传文档、添加网络资源、提出问题，并实时查看智能体的思考流程。

## 功能特点

### 1. 交互式知识库管理

* 上传文档以拓展知识库范围

* 动态添加 URL 以纳入网络内容

* 采用 LanceDB 实现向量数据库持久化存储

### 2. 透明化推理过程

* 实时展示智能体的思考步骤

* 推理过程与最终答案**并列视图**，清晰对比

* 完整呈现 RAG 流程，无黑箱操作

### 3. 高级 RAG 能力

* 基于 OpenAI 嵌入模型的向量搜索，实现语义匹配

* 支持来源归因，附带引用标注，确保信息可追溯

## 智能体配置

* 语言处理：Claude 3.5 Sonnet 模型

* 向量搜索：OpenAI 嵌入模型

* 推理分析：ReasoningTools 工具（用于分步推理）

* 自定义能力：支持个性化配置智能体指令

## 前置条件

需准备以下 API 密钥：

### 1. Anthropic API 密钥

* 注册地址：[console.anthropic.com](https://console.anthropic.com)

* 操作步骤：进入 “API Keys” 板块，创建新的 API 密钥

### 2. OpenAI API 密钥

* 注册地址：[platform.openai.com](https://platform.openai.com)

* 操作步骤：进入 “API Keys” 板块，生成新的 API 密钥

## 运行步骤

### 1. 克隆仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd rag\_tutorials/agentic\_rag\_with\_reasoning
```

### 2. 安装依赖包

```
pip install -r requirements.txt
```

### 3. 启动应用

```
streamlit run rag\_reasoning\_agent.py
```

### 4. 配置 API 密钥

* 在第一个输入框中输入 Anthropic API 密钥

* 在第二个输入框中输入 OpenAI API 密钥

* 注：两个密钥均为必填项，缺一不可

### 5. 使用应用

* 添加知识来源：通过侧边栏添加 URL 至知识库

* 提出问题：在主输入框中输入查询内容

* 查看推理：实时观察智能体的思考过程

* 获取答案：接收包含来源引用的完整回复

## 工作原理

该应用采用先进的 RAG 流水线架构，核心流程如下：

### 知识库搭建

1. 利用 WebBaseLoader 从 URL 加载文档内容

2. 对文本进行分块处理，并通过 OpenAI 嵌入模型生成向量

3. 将向量存储到 LanceDB 中，实现高效检索

4. 通过向量搜索完成相关信息的语义匹配

### 智能体处理

1. 用户查询触发智能体的推理流程

2. ReasoningTools 工具辅助智能体实现分步思考

3. 智能体在知识库中检索相关信息

4. Claude 4 Sonnet 模型生成包含引用的完整答案

### 界面流程

* 输入 API 密钥 → 添加知识来源 → 提出问题

* 推理过程与答案生成**并列展示**，同步更新

* 附带来源引用，确保透明度与可验证性
