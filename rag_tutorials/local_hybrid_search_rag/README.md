# 🖥️ 支持混合搜索的本地RAG应用

## 🖥️ 支持混合搜索的本地 RAG 应用

这是一款功能强大的文档问答应用，依托混合搜索（RAG，检索增强生成）技术与本地大语言模型（LLM），提供全面精准的问答结果。应用基于 RAGLite 构建，可实现可靠的文档处理与检索；同时采用 Streamlit 搭建直观的聊天界面，将文档专属知识与本地 LLM 能力深度结合，最终输出准确且贴合上下文的响应。

## 演示示例：

[https://github.com/user-attachments/assets/375da089-1ab9-4bf4-b6f3-733f44e47403](https://github.com/user-attachments/assets/375da089-1ab9-4bf4-b6f3-733f44e47403)

## 快速开始

若需立即测试，可使用以下经过验证的模型配置（兼顾性能与资源占用，在配备 8GB 内存的 MacBook Air M2 上可流畅运行）：

```
\# LLM模型（大语言模型）

bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4\_K\_M.gguf@4096

\# 嵌入模型（用于文本向量转换）

lm-kit/bge-m3-gguf/bge-m3-Q4\_K\_M.gguf@1024
```

## 功能特性

### 1. 本地 LLM 集成

* 采用`llama-cpp-python`框架加载本地模型进行推理

* 支持多种量化格式（推荐使用 Q4\_K\_M 格式，平衡速度与精度）

* 可配置上下文窗口大小，适配不同场景需求

### 2. 文档处理

* 支持 PDF 文档上传与解析处理

* 自动完成文本分块（拆分长文本为适合模型处理的片段）与嵌入（转换文本为向量）

* 结合**语义匹配**与**关键词匹配**的混合搜索方式，提升检索全面性

* 通过重排序（Reranking）优化上下文选择，筛选最相关的内容用于生成答案

### 3. 多模型协同

* 本地 LLM 负责文本生成（如 Llama-3.2-3B-Instruct 模型）

* 本地嵌入模型（如 BGE 系列模型）生成文本向量

* 本地重排序工具 FlashRank 优化检索结果排序

## 前置准备

### 1. 安装 spaCy 模型（文本处理依赖）

```
pip install https://github.com/explosion/spacy-models/releases/download/xx\_sent\_ud\_sm-3.7.0/xx\_sent\_ud\_sm-3.7.0-py3-none-any.whl
```

### 2. 安装加速版`llama-cpp-python`（可选但推荐，提升推理速度）

```
\# 配置安装参数

LLAMA\_CPP\_PYTHON\_VERSION=0.3.2  # 模型版本

PYTHON\_VERSION=310              # Python版本（3.10填310，3.11填311，3.12填312）

ACCELERATOR=metal               # 加速方式：Mac选metal；NVIDIA显卡选cu121

PLATFORM=macosx\_11\_0\_arm64      # 系统平台：Mac选此值；Linux选linux\_x86\_64；Windows选win\_amd64

\# 安装加速版本

pip install "https://github.com/abetlen/llama-cpp-python/releases/download/v\$LLAMA\_CPP\_PYTHON\_VERSION-\$ACCELERATOR/llama\_cpp\_python-\$LLAMA\_CPP\_PYTHON\_VERSION-cp\$PYTHON\_VERSION-cp\$PYTHON\_VERSION-\$PLATFORM.whl"
```

### 3. 安装项目依赖

```
\# 克隆代码仓库

git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

\# 进入项目目录

cd awesome-llm-apps/rag\_tutorials/local\_hybrid\_search\_rag

\# 安装依赖包

pip install -r requirements.txt
```

## 模型配置

RAGLite 扩展了 LiteLLM 的功能，支持通过`llama-cpp-python`加载 llama.cpp 格式的模型。若需选择模型（如从 bartowski 仓库获取），需使用特定格式的模型标识符：`llama-cpp-python/< Hugging Face仓库ID >/< 文件名 >@< 可选参数 >`，其中可选参数可指定模型的上下文长度或向量维度。

### 1. LLM 模型路径格式

```
llama-cpp-python/<仓库名>/<模型目录>/<文件名>@<上下文长度>
```

示例：

```
bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4\_K\_M.gguf@4096
```

### 2. 嵌入模型路径格式

```
llama-cpp-python/<仓库名>/<模型目录>/<文件名>@<向量维度>
```

示例：

```
lm-kit/bge-m3-gguf/bge-m3-Q4\_K\_M.gguf@1024
```

## 数据库配置

应用支持多种数据库后端，用于存储文本向量与文档数据：

### PostgreSQL（推荐）

1. 访问[N](https://neon.tech)[e](https://neon.tech)[on 平台](https://neon.tech)，一键创建免费的无服务器（Serverless）PostgreSQL 数据库

2. 支持即时配置与 “缩容至零”（scale-to-zero）能力（无流量时自动释放资源，降低成本）

3. 连接字符串格式：

```
postgresql://用户名:密码@ep-xyz.区域.aws.neon.tech/数据库名
```
## 运行步骤

### 1. 启动应用

```
streamlit run local\_main.py
```

### 2. 配置应用参数

* 在界面中输入 LLM 模型路径（按上述 “LLM 模型路径格式” 填写）

* 输入嵌入模型路径（按上述 “嵌入模型路径格式” 填写）

* 设置数据库 URL（如使用 PostgreSQL，填写上述连接字符串；默认可使用本地 SQLite）

* 点击 “保存配置”（Save Configuration）

### 3. 上传文档

* 通过界面上传 PDF 文件

* 等待文档处理完成（含分块、嵌入、存储步骤）

### 4. 开始问答

* 针对已上传的文档提问

* 应用将通过本地 LLM 生成答案

* 若问题与文档无关，将自动切换为通用知识问答模式

## 注意事项

* 多数场景推荐使用 4096 的上下文窗口大小，兼顾处理能力与资源消耗

* Q4\_K\_M 量化格式在速度与生成质量间平衡最优，适合本地部署

* 1024 维度的 BGE-M3 嵌入模型适配性最强，推荐优先使用

* 本地模型运行需足够的内存（RAM）与 CPU/GPU 资源，低配置设备可能出现卡顿

* 加速支持：Mac 设备可启用 Metal 加速，NVIDIA 显卡设备可启用 CUDA 加速

## 贡献指南

欢迎各位开发者贡献代码！如有优化建议或功能改进，可直接提交拉取请求（Pull Request）。
