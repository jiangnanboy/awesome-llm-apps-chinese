# 🐋 Qwen 3 本地RAG推理智能体

## 🐋 Qwen 3 本地 RAG 推理智能体

本 RAG 应用展示了如何通过 Ollama 调用本地运行的 Qwen 3 与 Gemma 3 模型，构建功能强大的检索增强生成（RAG）系统。它融合了文档处理、向量搜索与网页搜索能力，能为用户查询提供准确且贴合上下文的响应。

## 功能特点

### 🧠 多种本地 LLM 选择

* Qwen3（1.7b、8b 版本）—— 阿里巴巴最新推出的语言模型

* Gemma3（1b、4b 版本）—— 谷歌推出的高效语言模型，支持多模态能力

* DeepSeek（1.5b 版本）—— 可选的替代模型

### 📚 全面的 RAG 系统

* 上传并处理 PDF 文档

* 从网页 URL 提取内容

* 智能文本分块与嵌入处理

* 支持调整阈值的相似度搜索

### 🌐 网页搜索集成

* 当文档知识不足以回答问题时，自动切换至网页搜索

* 可配置域名过滤规则

* 响应中会标注信息来源

### 🔄 灵活的操作模式

* 可在 “RAG 模式” 与 “直接 LLM 交互模式” 之间切换

* 可按需强制启用网页搜索

* 可调整文档检索的相似度阈值

### 💾 向量数据库集成

* 采用 Qdrant 向量数据库，实现高效的相似度搜索

* 文档嵌入结果支持持久化存储

## 如何开始使用

### 前置条件

* 本地已安装[Ollama](https://ollama.ai/)

* Python 3.8 及以上版本

* Qdrant 账号（提供免费层级，用于向量存储）

* Exa API 密钥（可选，用于启用网页搜索功能）

### 安装步骤

1. 克隆 GitHub 仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd rag\_tutorials/qwen\_local\_rag
```

2. 安装所需依赖：

```
pip install -r requirements.txt
```

3. 通过 Ollama 拉取所需模型：

```
ollama pull qwen3:1.7b # 也可选择其他你想使用的模型

ollama pull snowflake-arctic-embed # 也可选择其他你想使用的嵌入模型
```

4. 通过 Docker 在本地运行 Qdrant：

```
docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334 \\

&#x20;   -v "\$(pwd)/qdrant\_storage:/qdrant/storage:z" \\

&#x20;   qdrant/qdrant
```

5. 获取 API 密钥：

* Exa API 密钥（可选，用于网页搜索）

6. 运行应用：

```
streamlit run qwen\_local\_rag\_agent.py
```

## 工作原理

### 1. 文档处理

* 使用 PyPDFLoader 处理 PDF 文件

* 通过 WebBaseLoader 提取网页内容

* 借助 RecursiveCharacterTextSplitter 将文档拆分为文本块

### 2. 向量数据库

* 利用 Ollama 的嵌入模型，将文档文本块转换为嵌入向量

* 嵌入向量存储到 Qdrant 向量数据库中

* 通过相似度搜索，根据用户查询检索相关文档

### 3. 查询处理

* 分析用户查询，确定最优信息来源

* 基于相似度阈值判断文档与查询的相关性

* 若未找到相关文档，自动切换至网页搜索

### 4. 响应生成

* 本地 LLM（Qwen/Gemma）结合检索到的上下文生成响应

* 向用户展示信息来源标注

* 若使用了网页搜索结果，会明确标识

## 配置选项

* **模型选择**：在不同版本的 Qwen、Gemma 与 DeepSeek 模型间切换

* **RAG 模式**：开启 / 关闭 RAG 功能，切换至直接与 LLM 交互的模式

* **搜索调优**：调整文档检索的相似度阈值

* **网页搜索**：启用 / 禁用网页搜索 fallback 功能，配置域名过滤规则

## 适用场景

* **文档问答**：针对已上传的文档提问，获取精准答案

* **研究助手**：结合文档知识与网页搜索，辅助研究工作

* **本地隐私保护**：处理敏感文档时，无需将数据发送至外部 API

* **离线运行**：在网络有限或无网络环境下，仍能使用先进的 AI 功能

## 依赖要求

完整的依赖列表请参考 `requirements.txt` 文件。
