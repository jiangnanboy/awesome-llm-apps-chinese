# 本地 RAG 智能体

## 🦙 基于 Llama 3.2 的本地 RAG 智能体

本应用实现了一个检索增强生成（RAG）系统，通过 Ollama 调用 Llama 3.2 模型，并使用 Qdrant 作为向量数据库。

### 功能特点

* 完全本地化的 RAG 实现

* 通过 Ollama 驱动 Llama 3.2 模型

* 使用 Qdrant 进行向量搜索

* 交互式操作界面

* 无外部 API 依赖

### 如何开始使用？

1. 克隆 GitHub 仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
```

2. 安装所需依赖：

```
cd awesome-llm-apps/rag\_tutorials/local\_rag\_agent

pip install -r requirements.txt
```

3. 本地安装并启动[Qdrant](https://qdrant.tech/)向量数据库

```
docker pull qdrant/qdrant

docker run -p 6333:6333 qdrant/qdrant
```

4. 安装[Ollama](https://ollama.com/download)并拉取 Llama 3.2（作为大语言模型）和 OpenHermes（作为 OllamaEmbedder 的嵌入模型）

```
ollama pull llama3.2

ollama pull openhermes
```

5. 运行 AI RAG 智能体

```
python local\_rag\_agent.py
```

6. 打开网页浏览器，导航到控制台输出中提供的 URL，通过操作界面与 RAG 智能体进行交互。
