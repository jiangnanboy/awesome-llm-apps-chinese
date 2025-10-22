# 🐋 Deepseek 本地 RAG 推理代理（Deepseek Local RAG Reasoning Agent）

一款将本地 Deepseek 模型与 RAG（检索增强生成）能力相结合的强大推理代理。该应用基于 Deepseek（通过 Ollama 实现）、Snowflake（用于嵌入向量生成）、Qdrant（用于向量存储）和 Agno（用于代理编排）构建，既支持简单的本地对话功能，也能提供增强型 RAG 交互 —— 具备全面的文档处理与网页搜索能力。

## 功能特点（Features）

* **双运行模式（Dual Operation Modes）**

  * 本地对话模式（Local Chat Mode）：在本地环境中与 Deepseek 模型直接交互

  * RAG 模式（RAG Mode）：结合文档上下文与网页搜索功能实现增强推理（基于 llama3.2 模型）

* **文档处理（Document Processing）**（RAG 模式专用）

  * PDF 文档上传与处理

  * 网页内容提取

  * 文本自动分块与嵌入向量生成

  * 在 Qdrant 云服务中存储向量数据

* **智能查询（Intelligent Querying）**（RAG 模式专用）

  * 基于 RAG 的文档检索

  * 带阈值过滤的相似度搜索

  * 自动降级至网页搜索（当文档检索结果不足时）

  * 回答内容的来源标注

* **高级功能（Advanced Capabilities）**

  * 集成 Exa AI 网页搜索

  * 网页搜索的自定义域名过滤

  * 上下文感知式响应生成

  * 对话历史管理

  * 推理过程可视化

* **模型专属功能（Model Specific Features）**

  * 灵活的模型选择：

    * Deepseek r1 1.5b（轻量型，适用于大多数笔记本电脑）

    * Deepseek r1 7b（功能更强，需更优硬件支持）

  * 采用 Snowflake Arctic 嵌入模型（SOTA，即当前最优技术水平）生成向量嵌入

  * 基于 Agno Agent 框架实现代理编排

  * 基于 Streamlit 构建的交互式界面

## 前置条件（Prerequisites）

### 1. Ollama 环境配置（Ollama Setup）

1. 安装 Ollama（官网链接：[https://ollama.ai](https://ollama.ai)）

2. 拉取（下载）Deepseek r1 模型（可选择以下任一或全部）：

```
\# 拉取轻量型模型

ollama pull deepseek-r1:1.5b

\# 拉取功能增强型模型（需硬件支持）

ollama pull deepseek-r1:7b

\# 拉取其他依赖模型

ollama pull snowflake-arctic-embed

ollama pull llama3.2
```

### 2. Qdrant 云服务配置（Qdrant Cloud Setup）（RAG 模式专用）

1. 访问 Qdrant 云服务官网（链接：[https://cloud.qdrant.io/](https://cloud.qdrant.io/)）

2. 注册账号或登录已有账号

3. 创建新的集群（cluster）

4. 获取认证信息：

* Qdrant API 密钥（Qdrant API Key）：在 “API Keys” 栏目中查找

* Qdrant 地址（Qdrant URL）：集群专属地址（格式示例：`https://xxx-xxx.cloud.qdrant.io`）

### 3. Exa AI API 密钥（Exa AI API Key）（可选）

1. 访问 Exa AI 官网（链接：[https://exa.ai](https://exa.ai)）

2. 注册账号

3. 生成 API 密钥（用于启用网页搜索功能）

## 运行步骤（How to Run）

1. 克隆代码仓库：

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd rag\_tutorials/deepseek\_local\_rag\_agent
```
2. 安装依赖包：

```
pip install -r requirements.txt
```

3. 启动应用：

```
streamlit run deepseek\_rag\_agent.py
```

### 术语说明（补充）

* **RAG（Retrieval-Augmented Generation）**：检索增强生成，一种结合外部知识库检索与语言模型生成的技术，可提升回答的准确性与时效性。

* **向量嵌入（Embedding）**：将文本等非结构化数据转换为数值向量的过程，便于计算机进行相似度计算与检索。

* **Streamlit**：一款用于快速构建数据科学与机器学习 Web 应用的 Python 框架，无需复杂前端开发。

* **Ollama**：一款简化本地大语言模型部署与管理的工具，支持一键拉取、运行多种开源模型。
