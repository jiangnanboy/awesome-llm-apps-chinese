# 🧬 多模态人工智能代理（Multimodal AI Agent）

这是一款基于 Streamlit 框架开发的应用程序，借助谷歌（Google）的 Gemini 2.0 模型，整合了视频分析与网页搜索功能。该代理能够对上传的视频进行分析，并结合视觉理解与网页搜索能力来回答问题。

## 功能特点（Features）

* 采用 Gemini 2.0 Flash 模型进行视频分析

* 通过 DuckDuckGo（达鸭搜索）集成网页搜索功能

* 支持多种视频格式（MP4、MOV、AVI）

* 实时视频处理

* 融合视觉与文本的综合分析

## 如何开始使用？（How to get Started?）

1. 克隆 GitHub 代码仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd ai\_agent\_tutorials/multimodal\_ai\_agent
```

2. 安装所需依赖包：

```
pip install -r requirements.txt
```
3. 获取谷歌 Gemini API 密钥

* 注册[Google AI Studio 账号](https://aistudio.google.com/apikey)，并获取 API 密钥。

4. 将 Gemini API 密钥配置为环境变量

```
GOOGLE\_API\_KEY=your\_api\_key\_here  # 此处替换为你的实际API密钥
```

1. 运行 Streamlit 应用

```
streamlit run multimodal\_agent.py
```

### 补充说明

* **Streamlit**：一款用于快速构建数据科学与机器学习 Web 应用的 Python 框架，无需复杂的前端开发知识。

* **Gemini 2.0**：谷歌推出的多模态大语言模型，支持文本、图像、音频、视频等多种输入类型的理解与处理，其中 “Flash” 版本侧重高效快速的推理。

* **DuckDuckGo**：一款注重用户隐私保护的搜索引擎，与谷歌搜索相比，不会追踪用户搜索行为，常用于对隐私要求较高的开发场景。
