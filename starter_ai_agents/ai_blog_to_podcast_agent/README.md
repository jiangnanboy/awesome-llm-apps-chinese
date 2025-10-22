# 📰 ➡️ 🎙️ 博客转播客工具（Blog to Podcast Agent）

这是一款基于 Streamlit 框架开发的应用程序，可帮助用户将任意博客文章转换为播客。该应用整合了三大核心技术：采用 OpenAI 的 GPT-4 模型进行内容总结、借助 Firecrawl 抓取博客内容、通过 ElevenLabs API 生成音频。用户只需输入博客的 URL 链接，应用就能基于该博客内容生成一期播客节目。

## 功能特点（Features）

* **博客内容抓取（Blog Scraping）**：通过 Firecrawl API 抓取任意公开博客 URL 的完整内容。

* **摘要生成（Summary Generation）**：利用 OpenAI GPT-4 模型，生成生动简洁的博客摘要（字符数控制在 2000 字以内）。

* **播客生成（Podcast Generation）**：通过 ElevenLabs 语音 API，将生成的摘要转换为音频播客。

* **API 密钥集成（API Key Integration）**：应用运行需依赖 OpenAI、Firecrawl 和 ElevenLabs 的 API 密钥，用户可通过侧边栏安全输入密钥。

## 搭建步骤（Setup）

### 必备条件（Requirements）

1. **API 密钥（API Keys）**：

* **OpenAI API 密钥**：在 OpenAI 平台注册账号，即可获取个人 API 密钥。

* **ElevenLabs API 密钥**：前往 ElevenLabs 平台，获取对应的 API 密钥。

* **Firecrawl API 密钥**：访问 Firecrawl 平台，获取专属 API 密钥。

2. **Python 3.8 及以上版本**：确保设备已安装 Python 3.8 或更高版本。

### 安装流程（Installation）

1. 克隆本代码仓库：

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps

cd ai\_agent\_tutorials/ai\_blog\_to\_podcast\_agent
```

2. 安装所需的 Python 依赖包：

```
pip install -r requirements.txt
```

### 运行应用（Running the App）

1. 启动 Streamlit 应用：

```
streamlit run blog\_to\_podcast\_agent.py
```

2. 在应用界面中操作：

* 在侧边栏输入你的 OpenAI、ElevenLabs 和 Firecrawl API 密钥。

* 输入你想要转换的博客 URL 链接。

* 点击 “🎙️ 生成播客（Generate Podcast）” 按钮。

* 即可播放生成的播客，也可选择下载播客音频。
