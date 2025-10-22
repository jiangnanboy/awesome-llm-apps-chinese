# 🛫 人工智能旅行顾问（AI Travel Agent）

这款 Streamlit 应用是一款基于人工智能的旅行顾问，它借助 OpenAI GPT-4o 生成个性化旅行行程。该应用能自动完成你理想假期的调研、规划与整理流程，让你轻松探索各类热门目的地。

## 功能特点（Features）

* 调研并发现热门旅行目的地、活动及住宿选择

* 根据你的旅行天数自定义行程安排

* 借助 GPT-4o 的强大能力，生成智能且个性化的旅行计划

* 将行程下载为日历（.ics 格式）文件，以便导入谷歌日历、苹果日历或其他日历应用

## 如何开始使用？（How to get Started?）

1. 克隆 GitHub 仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd awesome-llm-apps/ai\_agent\_tutorials/ai\_travel\_agent
```

2. 安装所需依赖：

```
pip install -r requirements.txt
```

3. 获取你的 OpenAI API 密钥

* 注册一个[OpenAI 账户](https://platform.openai.com/)（或你选择的大语言模型服务商账户），并获取 API 密钥。

4. 获取你的 SerpAPI 密钥

* 注册一个[SerpAPI 账户](https://serpapi.com/)，并获取 API 密钥。

5. 运行 Streamlit 应用

```
streamlit run travel\_agent.py
```

若需使用本地大语言模型（通过 Ollama）：

```
streamlit run local\_travel\_agent.py
```

## 工作原理（How it Works?）

人工智能旅行顾问包含两个核心模块：

* **调研模块（Researcher）**：根据用户指定的目的地和旅行时长生成搜索关键词，并通过 SerpAPI 在网络上搜索相关活动与住宿信息。

* **规划模块（Planner）**：结合调研结果与用户偏好，生成个性化行程草案，其中包含推荐活动、餐饮选择及住宿建议。

## 日历下载功能使用方法（Using the Calendar Download Feature）

生成旅行行程后，按以下步骤操作：

1. 点击 “生成行程”（Generate Itinerary）按钮旁显示的 “将行程下载为日历（.ics 格式）”（Download Itinerary as Calendar (.ics)）按钮

2. 将.ics 文件保存到你的电脑中

3. 将该文件导入你常用的日历应用（谷歌日历、苹果日历、Outlook 等）

4. 行程中的每一天将以 “全天事件” 形式显示在你的日历中

5. 每天活动的完整详情会包含在事件描述中

该功能便于你随时查看旅行计划，即使在离线状态下，也能在所有设备上获取行程信息。

## 本地版本与云端版本对比（Local vs Cloud Version）

* **travel\_agent.py**：使用 OpenAI 的 GPT-4o 生成高质量行程（需 OpenAI API 密钥）

* **local\_travel\_agent.py**：通过 Ollama 实现本地大语言模型推理，无需将数据发送至外部 API（需安装并运行 Ollama）
