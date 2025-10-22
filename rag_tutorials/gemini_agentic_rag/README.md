# 🤔 基于 Gemini Flash Thinking 的智能体 RAG（Agentic RAG with Gemini Flash Thinking）

一个基于全新 Gemini 2.0 Flash Thinking 模型和 gemini-exp-1206 构建的 RAG 智能体系统，结合 Qdrant 向量存储和 Agno（原 phidata）智能体编排框架。该应用具备智能查询重写、文档处理和网页搜索降级能力，可提供全面的 AI 驱动响应。

## 功能特点（Features）

* **文档处理（Document Processing）**

  * PDF 文档上传与处理

  * 网页内容提取

  * 文本自动分块与嵌入生成

  * Qdrant 云端向量存储

* **智能查询（Intelligent Querying）**

  * 为优化检索效果的查询重写

  * 基于 RAG 的文档检索

  * 带阈值过滤的相似度搜索

  * 自动降级至网页搜索（检索结果不足时）

  * 回答内容的来源标注

* **高级功能（Advanced Capabilities）**

  * 集成 Exa AI 网页搜索

  * 网页搜索的自定义域名过滤

  * 上下文感知的响应生成

  * 对话历史管理

  * 查询重构智能体

* **模型专属功能（Model Specific Features）**

  * Gemini Thinking 2.0 Flash 用于对话与推理

  * Gemini 嵌入模型用于生成向量嵌入

  * Agno 智能体框架用于编排

  * 基于 Streamlit 的交互式界面

## 前置条件（Prerequisites）

### 1. Google API 密钥（Google API Key）

1. 访问[Google AI Studio](https://aistudio.google.com/apikey)

2. 注册或登录账号

3. 创建新的 API 密钥

### 2. Qdrant 云服务配置（Qdrant Cloud Setup）

1. 访问[Qdrant Cloud](https://cloud.qdrant.io/)

2. 注册或登录账号

3. 创建新集群

4. 获取凭证信息：

* Qdrant API 密钥：在 API Keys 栏目中获取

* Qdrant URL：集群专属地址（格式：`https://xxx-xxx.cloud.qdrant.io`）

### 3. Exa AI API 密钥（可选）

1. 访问[Exa AI](https://exa.ai)

2. 注册账号

3. 生成用于网页搜索的 API 密钥

## 运行步骤（How to Run）

1. 克隆代码仓库：

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd rag\_tutorials/gemini\_agentic\_rag
```

2. 安装依赖：

```
pip install -r requirements.txt
```
3. 启动应用：

```
streamlit run agentic\_rag\_gemini.py
```

## 使用说明（Usage）

1. 在侧边栏配置 API 密钥：

* 输入 Google API 密钥

* 添加 Qdrant 凭证

* （可选）添加 Exa AI 密钥以启用网页搜索

2. 上传文档：

* 使用文件上传器上传 PDF

* 输入 URL 以提取网页内容

3. 提问交互：

* 在聊天界面输入查询

* 查看重写后的查询及来源

* 相关时可查看网页搜索结果

4. 会话管理：

* 按需清除聊天历史

* 配置网页搜索域名

* 监控已处理的文档

### 补充说明

* **Agentic RAG**：具备智能体能力的检索增强生成系统，相比传统 RAG 增加了自主决策、任务分解和工具调用能力

* **Gemini Flash Thinking**：Google 推出的具备快速推理能力的大语言模型版本，擅长高效处理复杂任务

* **查询重写（Query Rewriting）**：系统自动优化用户查询表述，以提升检索准确性的技术

* **Agno**：原 phidata 框架，专注于构建可扩展的 AI 智能体应用，简化复杂工作流编排
