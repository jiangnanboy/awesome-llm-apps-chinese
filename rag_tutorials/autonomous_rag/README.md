# 🤖 AutoRAG：基于 GPT-4o 和向量数据库的自主检索增强生成系统

这款 Streamlit 应用采用 OpenAI 的 GPT-4o 模型和 PgVector 数据库，实现了一个自主检索增强生成（RAG）系统。用户可以上传 PDF 文档并将其添加到知识库中，然后结合知识库内容和网络搜索结果向 AI 助手提问。

### 功能特点

* 与 AI 助手交互的聊天界面

* PDF 文档上传与处理

* 基于 PostgreSQL 和 Pgvector 的知识库集成

* 借助 DuckDuckGo 实现的网络搜索功能

* 助手数据和对话的持久化存储

### 如何开始使用？

1. 克隆 GitHub 仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd awesome-llm-apps/rag\_tutorials/autonomous\_rag
```

2. 安装所需依赖：

```
pip install -r requirements.txt
```

3. 确保 PgVector 数据库正在运行：

   该应用默认 PgVector 运行在[localhost:5532](http://localhost:5532/)。如果您的配置不同，请调整代码中的相关设置。

```
docker run -d \\

&#x20; -e POSTGRES\_DB=ai \\

&#x20; -e POSTGRES\_USER=ai \\

&#x20; -e POSTGRES\_PASSWORD=ai \\

&#x20; -e PGDATA=/var/lib/postgresql/data/pgdata \\

&#x20; -v pgvolume:/var/lib/postgresql/data \\

&#x20; -p 5532:5432 \\

&#x20; \--name pgvector \\

&#x20; phidata/pgvector:16
```

4. 运行 Streamlit 应用

```
streamlit run autorag.py
```
