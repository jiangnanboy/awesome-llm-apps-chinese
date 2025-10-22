# 👀 具备混合搜索功能的 RAG 应用程序

这是一款功能强大的文档问答应用程序，它借助混合搜索（RAG，检索增强生成）技术和 Claude 的先进语言处理能力，提供全面的问答结果。该系统基于 RAGLite 构建，可实现可靠的文档处理与检索；同时采用 Streamlit 搭建直观的聊天界面，将文档专属知识与 Claude 的通用智能无缝结合，从而输出准确且贴合上下文的响应。

## 功能特性

* **混合搜索问答**

  * 基于 RAG 技术，为文档相关查询提供精准答案

  * 对于通用知识类问题，自动切换至 Claude 进行解答

* **文档处理**：

  * 支持 PDF 文档上传与处理

  * 自动完成文本分块与嵌入（将文本转换为机器可识别的向量格式）

  * 结合语义匹配与关键词匹配的混合搜索方式

  * 通过重排序（Reranking）优化上下文选择，提升答案相关性

* **多模型集成**：

  * 文本生成：采用 Claude（已通过 Claude 3 Opus 版本测试）

  * 嵌入生成：采用 OpenAI（已通过 text-embedding-3-large 模型测试）

  * 重排序：采用 Cohere（已通过 Cohere 3.5 重排序模型测试）

## 前置条件

您需要准备以下 API 密钥，并完成数据库配置：

1. **数据库**：在[Neon](https://neon.tech)平台创建免费的 PostgreSQL 数据库，步骤如下：

* 访问 Neon 平台完成注册 / 登录

* 创建新项目

* 复制连接字符串（格式示例：`postgresql://user:pass@ep-xyz.region.aws.neon.tech/dbname`）

2. **API 密钥**：

* [OpenAI API 密钥](https://platform.openai.com/api-keys)（用于生成文本嵌入）

* [Anthropic API 密钥](https://console.anthropic.com/settings/keys)（用于调用 Claude 模型）

* [Cohere API 密钥](https://dashboard.cohere.com/api-keys)（用于文本重排序）

## 如何开始使用？

1. **克隆代码仓库**：

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd awesome-llm-apps/rag\_tutorials/hybrid\_search\_rag
```

2. **安装依赖包**：

```
pip install -r requirements.txt
```

3. **安装 spaCy 模型**：

```
pip install https://github.com/explosion/spacy-models/releases/download/xx\_sent\_ud\_sm-3.7.0/xx\_sent\_ud\_sm-3.7.0-py3-none-any.whl
```

4. **运行应用程序**：

```
streamlit run main.py
```

## 使用说明

1. 启动应用程序

2. 在侧边栏输入以下 API 密钥：

* OpenAI API 密钥

* Anthropic API 密钥

* Cohere API 密钥

* 数据库 URL（可选，默认使用 SQLite 数据库）

1. 点击 “保存配置”（Save Configuration）

2. 上传 PDF 文档

3. 开始提问！

* 文档相关问题将通过 RAG 技术生成答案

* 通用问题将直接调用 Claude 模型解答

## 数据库选项

该应用程序支持多种数据库后端：

* **PostgreSQL（推荐）**：

  * 在[Neon](https://neon.tech)平台创建免费的无服务器（Serverless）PostgreSQL 数据库

  * 支持即时配置与 “缩容至零”（scale-to-zero）能力（无流量时自动停止资源占用，降低成本）

  * 连接字符串格式：`postgresql://user:pass@ep-xyz.region.aws.neon.tech/dbname`

* **MySQL**：

```
mysql://user:pass@host:port/db
```
* **SQLite（本地开发适用）**：

```
sqlite:///path/to/db.sqlite
```

## 贡献指南

欢迎大家贡献代码！如有改进建议，可随时提交拉取请求（Pull Request）。
