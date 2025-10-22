# 🔄 纠错型 RAG 智能体（Corrective RAG Agent）

一种复杂的检索增强生成（Retrieval-Augmented Generation，简称 RAG）系统，基于 LangGraph 实现了多阶段纠错工作流。该系统融合了文档检索、相关性评分、查询转换和网络搜索功能，可提供全面且准确的响应。

## 功能特点（Features）

* **智能文档检索（Smart Document Retrieval）**：采用 Qdrant 向量数据库实现高效的文档检索

* **文档相关性评分（Document Relevance Grading）**：使用 Claude 3.5 Sonnet 模型评估文档相关性

* **查询转换（Query Transformation）**：在需要时优化查询语句，提升搜索结果质量

* **网络搜索备用（Web Search Fallback）**：当本地文档无法满足需求时，通过 Tavily API 进行网络搜索

* **多模型协同（Multi-Model Approach）**：结合 OpenAI 嵌入模型与 Claude 3.5 Sonnet 模型，分别处理不同任务

* **交互式界面（Interactive UI）**：基于 Streamlit 构建，支持便捷的文档上传与查询操作

## 运行步骤（How to Run?）

1. **克隆代码仓库（Clone the Repository）**：

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd rag\_tutorials/corrective\_rag
```

2. **安装依赖包（Install Dependencies）**：

```
pip install -r requirements.txt
```

3. **配置 API 密钥（Set Up API Keys）**：

   需获取以下 API 密钥：

* [OpenAI API 密钥](https://platform.openai.com/api-keys)（用于生成文档嵌入向量）

* [Anthropic API 密钥](https://console.anthropic.com/settings/keys)（用于调用 Claude 3.5 Sonnet 大语言模型）

* [Tavily API 密钥](https://app.tavily.com/home)（用于网络搜索）

* Qdrant 云服务配置（Qdrant Cloud Setup）：

1. 访问[Qdrant 云服务平台](https://cloud.qdrant.io/)

2. 创建账号或登录现有账号

3. 新建一个集群（cluster）

4. 获取认证信息：

* Qdrant API 密钥：在 “API Keys” 板块中查看

* Qdrant 链接（URL）：集群专属链接（格式：`https://xxx-xxx.aws.cloud.qdrant.io`）

1. **启动应用程序（Run the Application）**：

```
streamlit run corrective\_rag.py
```

2**使用应用程序（Use the Application）**：

* 上传文档或提供文档 URL

* 在查询框中输入问题

* 查看纠错型 RAG 的分步处理过程

* 获取全面的问题答案

## 技术栈（Tech Stack）

* **LangChain**：用于 RAG 流程编排与链路管理

* **LangGraph**：用于工作流（ workflow ）管理

* **Qdrant**：用于文档存储的向量数据库

* **Claude 3.5 Sonnet**：用于分析与生成任务的主语言模型

* **OpenAI**：用于生成文档嵌入向量

* **Tavily**：提供网络搜索能力

* **Streamlit**：用于构建用户界面（UI）
