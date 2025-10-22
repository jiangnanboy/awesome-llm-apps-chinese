# 🥸 人工智能表情包生成代理 - 浏览器端使用

人工智能表情包生成代理（AI Meme Generator Agent）是一款功能强大的浏览器自动化工具，可通过人工智能代理创建表情包。该应用将多大型语言模型（multi-LLM）能力与自动化浏览器交互相结合，通过直接操控网站，根据文本提示生成表情包。

## 功能特点

* **多大型语言模型支持（Multi-LLM Support）**

  * Claude 3.5 Sonnet（Anthropic 公司开发）

  * GPT-4o（OpenAI 公司开发）

  * Deepseek v3（深度求索（Deepseek）公司开发）

  * 支持基于 API 密钥验证的自动模型切换

* **浏览器自动化（Browser Automation）**：

  * 与 [imgflip.com](https://imgflip.com) 表情包模板直接交互

  * 自动搜索相关表情包格式

  * 为顶部 / 底部标题动态插入文本

  * 从生成的表情包中提取图片链接

* **智能生成流程（Smart Generation Workflow）**：

  * 从提示文本中提取动作动词

  * 隐喻性模板匹配

  * 多步骤质量验证

  * 针对生成失败的自动重试机制

* **用户友好界面（User-Friendly Interface）**：

  * 模型配置侧边栏

  * API 密钥管理功能

  * 带可点击链接的直接表情包预览

  * 响应式错误处理

需提供的 API 密钥：

* **Anthropic**（用于调用 Claude 模型）

* **Deepseek**（用于调用 Deepseek 模型）

* **OpenAI**（用于调用 GPT-4o 模型）

## 运行方法

1.**克隆代码仓库**：

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd ai\_agent\_tutorials/ai\_meme\_generator\_browseruse
```

2.**安装依赖包**：

```
pip install -r requirements.txt
```

如需安装 `playwright`，执行以下命令：

```
python -m playwright install --with-deps
```

3.**运行 Streamlit 应用**：

```
streamlit run ai\_meme\_generator\_agent.py
```

### 补充说明

* 文中 `LLM` 为 `Large Language Model`（大型语言模型）的缩写，是人工智能领域的核心技术之一，此处保留缩写以符合技术文档惯例。

* `Streamlit` 是一款用于快速构建数据科学和机器学习 Web 应用的 Python 框架，`playwright` 是微软开发的浏览器自动化工具，二者均为技术领域常用工具，保留原名以确保准确性。

* `imgflip.com` 是国外知名的表情包制作与分享平台，此处保留原网址以方便用户直接访问。
