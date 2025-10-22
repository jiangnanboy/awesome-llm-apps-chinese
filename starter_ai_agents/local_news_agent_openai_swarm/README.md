# 📰 多智能体人工智能新闻助手

这款 Streamlit 应用程序采用了复杂的新闻处理流程，通过多个专业的人工智能智能体（AI Agent）实现新闻文章的搜索、整合与总结功能。它借助 Ollama 调用 Llama 3.2 模型，并结合 DuckDuckGo 搜索引擎，提供全面的新闻分析服务。

## 功能特点

* 采用多智能体架构，各智能体分工明确：

  * 新闻搜索智能体（News Searcher）：查找最新新闻文章

  * 新闻整合智能体（News Synthesizer）：分析并整合信息

  * 新闻总结智能体（News Summarizer）：生成简洁、专业的摘要

* 基于 DuckDuckGo 的实时新闻搜索

* 生成符合美联社（AP）/ 路透社（Reuters）风格的摘要

* 简洁易用的 Streamlit 界面

## 如何开始使用？

1. 克隆 GitHub 仓库

```
git clone https://github.com/your-username/ai-news-processor.git

cd awesome-llm-apps/ai\_agent\_tutorials/local\_news\_agent\_openai\_swarm
```

2. 安装所需依赖包：

```
pip install -r requirements.txt
```

3. 通过 Ollama 拉取并运行 Llama 3.2 模型：

```
\# 拉取模型

ollama pull llama3.2

\# 验证安装

ollama list

\# 运行模型（可选测试步骤）

ollama run llama3.2
```

1. 创建包含配置信息的.env 文件：


```
OPENAI\_BASE\_URL=http://localhost:11434/v1

OPENAI\_API\_KEY=fake-key&#x20;
```

2. 运行 Streamlit 应用

```
streamlit run news\_agent.py
```

### 补充说明

* **技术术语解释**：

  * Streamlit：一款用于快速构建数据科学和机器学习 Web 应用的 Python 框架，无需复杂的前端开发即可创建交互式界面。

  * Ollama：一款轻量级的本地大模型运行工具，支持一键拉取、运行多种开源大模型（如 Llama、Gemini 等），无需复杂的环境配置。

  * .env 文件：用于存储环境变量（如 API 密钥、服务地址等）的配置文件，可避免敏感信息直接暴露在代码中，提升安全性。

* **命令说明**：代码块中以`#`开头的内容为注释，用于解释后续命令的功能，实际执行时无需输入注释部分。

