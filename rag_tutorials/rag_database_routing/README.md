# 📠 带数据库路由的 RAG 智能体（RAG Agent with Database Routing）

一款基于 Streamlit 开发的应用，演示了**检索增强生成（RAG）智能体**的高级实现方案，核心特点是具备智能查询路由功能。该系统整合了多个专业数据库，并搭配智能 fallback（降级）机制，确保能为用户查询提供可靠且准确的响应。

## 核心功能（Features）

* **文档上传（Document Upload）**：用户可上传与特定企业相关的多个 PDF 文档。这些文档会经过自动处理，并存储到以下三个专用数据库之一：

  * 产品信息库（Product Information）

  * 客户支持与常见问题库（Customer Support & FAQ）

  * 财务信息库（Financial Information）

* **自然语言查询（Natural Language Querying）**：用户可通过自然语言提问，系统会借助**Agno 智能体（路由专用智能体）** 自动将查询路由到最相关的数据库。

* **RAG 流程编排（RAG Orchestration）**：基于 LangChain 框架实现检索增强生成流程的编排，确保能精准检索出最相关的信息并呈现给用户。

* **降级机制（Fallback Mechanism）**：若在所有数据库中均未找到相关文档，系统会启动**LangGraph 智能体**（搭配 DuckDuckGo 搜索工具）执行网络检索，进而为用户提供答案。

## 运行步骤（How to Run?）

1. **克隆仓库（Clone the Repository）**：

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd rag\_tutorials/rag\_database\_routing  # 进入项目子目录
```

2. **安装依赖（Install Dependencies）**：

```
pip install -r requirements.txt  # 自动安装所有必要的Python库
```

3. **启动应用（Run the Application）**：

```
streamlit run rag\_database\_routing.py  # 启动Streamlit界面
```

4. **获取 OpenAI API 密钥（Get OpenAI API Key）**：

* 前往 OpenAI 官网申请 API 密钥

* 在应用中完成密钥配置（此步骤为必需，用于初始化应用中的语言模型）

5. **配置 Qdrant 云服务（Setup Qdrant Cloud）**：

* 访问 Qdrant 云服务官网：[https://cloud.qdrant.io/](https://cloud.qdrant.io/)

* 注册账号或登录已有账号

* 创建新的集群（Cluster）

* 获取认证信息（后续在应用中配置使用）：

  * Qdrant API 密钥：在「API Keys」栏目中获取

  * Qdrant 地址（URL）：集群专属地址（格式示例：[https://xxx-xxx.aws.cloud.qdrant.io](https://xxx-xxx.aws.cloud.qdrant.io)）

6. **上传文档（Upload Documents）**：通过应用中的「文档上传」模块，将 PDF 文档添加到目标数据库。

7. **提问查询（Ask Questions）**：在「查询输入」区域输入问题，应用会自动将查询路由到对应数据库并返回答案。

## 所用技术（Technologies Used）

* **LangChain**：用于 RAG 流程编排，确保信息检索与生成的高效性。

* **Agno 智能体（Agno Agent）**：作为路由智能体，负责判断查询对应的最相关数据库。

* **LangGraph 智能体（LangGraph Agent）**：作为降级机制的核心，在必要时调用 DuckDuckGo 执行网络检索。

* **Streamlit**：提供用户友好的界面，支持文档上传与查询交互。

* **Qdrant**：向量数据库，用于数据库管理，高效存储和检索文档嵌入向量（Embedding）。

## 工作原理（How It Works?）

### 1. 查询路由（Query Routing）

系统采用三级路由策略，确保查询精准匹配数据源：

* 第一级：跨所有数据库的向量相似度搜索（快速筛选候选数据库）

* 第二级：基于大语言模型（LLM）的路由（处理模糊或多意图查询）

* 第三级：网络搜索降级（针对数据库未覆盖的未知主题）

### 2. 文档处理（Document Processing）

* 自动从 PDF 中提取文本内容

* 带重叠窗口的智能文本分块（确保上下文连续性）

* 生成向量嵌入（将文本转换为机器可识别的向量格式）

* 高效存储到对应数据库

### 3. 答案生成（Answer Generation）

* 基于上下文的精准检索

* 多文档信息智能整合

* 带置信度的响应输出（标注答案可靠程度）

* 网络检索结果集成（补充数据库外的最新信息）
