# 基于 Claude 3.5 Sonnet 的 RAG 即服务

## 🖇️ 基于 Claude 3.5 Sonnet 的 RAG 即服务

使用 Claude 3.5 Sonnet 和 Ragie.ai 构建并部署生产级别的检索增强生成（RAG）服务。此实现能让你用不到 50 行 Python 代码，创建一个带有用户友好的 Streamlit 界面的文档查询系统。

### 功能特点

* 生产级别的 RAG 流水线

* 集成 Claude 3.5 Sonnet 用于响应生成

* 支持通过 URL 上传文档

* 实时文档查询

* 同时支持快速和精准两种文档处理模式

### 如何开始使用？

1. 克隆 GitHub 仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd awesome-llm-apps/rag\_tutorials/rag-as-a-service
```

2. 安装所需依赖：

```
pip install -r requirements.txt
```

3. 获取你的 Anthropic API 和 Ragie API 密钥

* 注册[Anthropic 账号](https://console.anthropic.com/)并获取 API 密钥

* 注册[Ragie 账号](https://www.ragie.ai/)并获取 API 密钥

4. 运行 Streamlit 应用

```
streamlit run rag\_app.py
```
