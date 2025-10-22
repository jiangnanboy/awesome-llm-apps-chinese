# 💻 网页抓取 AI 代理（Web Scrapping AI Agent）

这款 Streamlit 应用程序允许你借助 OpenAI API 和 scrapegraphai 库对网站进行抓取。只需提供你的 OpenAI API 密钥，输入想要抓取的网站 URL，并指定希望 AI 代理从该网站中提取的内容即可。

## 功能特点（Features）

* 只需提供网站 URL，即可对任意网站进行抓取

* 利用 OpenAI 的大语言模型（LLMs，如 GPT-3.5-turbo 或 GPT-4）实现智能抓取

* 可通过指定希望 AI 代理提取的内容，自定义抓取任务

## 如何开始使用？（How to get Started?）

1. 克隆 GitHub 仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd awesome-llm-apps/advanced\_tools\_frameworks/web\_scrapping\_ai\_agent
```

2. 安装所需依赖：

```
pip install -r requirements.txt
```

3. 获取你的 OpenAI API 密钥

* 注册一个[OpenAI 账户](https://platform.openai.com/)（或你选择的其他大语言模型提供商账户），并获取 API 密钥。

4. 运行 Streamlit 应用程序

```
streamlit run ai\_scrapper.py
```

## 工作原理（How it Works?）

1. 应用程序会提示你输入 OpenAI API 密钥，该密钥用于身份验证和访问 OpenAI 的语言模型。

2. 你可以为抓取任务选择所需的语言模型（GPT-3.5-turbo 或 GPT-4）。

3. 在提供的文本输入框中，输入你想要抓取的网站 URL。

4. 通过输入用户提示，指定希望 AI 代理从网站中提取的内容。

5. 应用程序会根据提供的 URL、用户提示和 OpenAI 配置，创建一个 SmartScraperGraph 对象。

6. SmartScraperGraph 对象会对网站进行抓取，并使用指定的语言模型提取所需信息。

7. 抓取结果会显示在应用程序中，供你查看。

### 补充说明

* **Streamlit**：一款用于快速构建数据科学和机器学习 Web 应用的 Python 框架，无需复杂的前端开发知识。

* **scrapegraphai**：专为结合大语言模型实现智能网页抓取设计的 Python 库，能更精准地识别和提取目标内容。

* **API 密钥安全**：OpenAI API 密钥涉及账户费用和权限，请勿在公共代码仓库或未授权场景中泄露。

