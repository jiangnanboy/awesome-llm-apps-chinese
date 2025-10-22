# ModelsLab 音乐生成器

这是一款基于 Streamlit 开发的应用程序，用户可借助 ModelsLab API（应用程序编程接口）和 OpenAI 的 GPT-4 模型生成音乐。用户只需输入描述性文本，说明自己想要生成的音乐类型，该应用程序就能根据所提供的文本提示，生成一首 MP3 格式的音乐曲目。

# 功能特点

* **音乐生成**：输入详细的音乐生成提示（包括音乐流派、使用乐器、情绪氛围等信息），应用程序将据此生成对应的音乐曲目。

* **MP3 格式输出**：生成的音乐以 MP3 格式保存，支持在线收听或下载。

* **用户友好界面**：采用简洁清晰的 Streamlit 界面设计，便于用户操作。

* **API 密钥集成**：应用程序运行需同时用到 OpenAI 和 ModelsLab 的 API 密钥。用户需在侧边栏输入 API 密钥以完成身份验证。

# 安装配置

## 必备条件

1. **API 密钥**：

* **OpenAI API 密钥**：前往 [OpenAI 官网](https://platform.openai.com/api-keys) 注册账号，即可获取个人 API 密钥。

* **ModelsLab API 密钥**：前往 [ModelsLab 官网](https://modelslab.com/dashboard/api-keys) 注册账号，即可获取个人 API 密钥。

2. **Python 3.8 及以上版本**：确保电脑中已安装 Python 3.8 或更高版本的 Python 环境。

## 安装步骤

1. 克隆本代码仓库：

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps

cd ai\_agent\_tutorials/ai\_models\_lab\_music\_generator\_agent
```

2. 安装所需的 Python 包：

```
pip install -r requirements.txt
```

## 运行应用程序

1. 启动 Streamlit 应用：

```
streamlit run models\_lab\_music\_generator\_agent.py
```

2. 在应用程序界面中操作：

* 输入音乐生成提示文本

* 点击 “生成音乐”（Generate Music）按钮

* 播放生成的音乐并下载。
