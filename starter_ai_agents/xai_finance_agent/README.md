# 📊 基于 xAI Grok 的人工智能财务代理（AI Finance Agent）

本应用程序借助 xAI 的 Grok 模型构建了一款财务分析代理，将实时股票数据与网页搜索功能相结合。它通过交互式操作界面（playground interface）提供结构化的财务洞察。

## 功能特点

* 由 xAI 的 Grok-beta 模型提供支持

* 通过 YFinance 实现实时股票数据分析

* 通过 DuckDuckGo 实现网页搜索功能

* 采用表格形式格式化呈现财务数据

* 具备交互式操作界面（playground interface）

## 如何开始使用？

1. 克隆 GitHub 代码仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd awesome-llm-apps/ai\_agent\_tutorials/xai\_finance\_agent
```

2. 安装所需依赖包：

```
cd awesome-llm-apps/ai\_agent\_tutorials/xai\_finance\_agent

pip install -r requirements.txt
```

3. 获取您的 OpenAI API 密钥

* 注册[xAI API 账户](https://console.x.ai/)

* 设置 XAI\_API\_KEY 环境变量：

```
export XAI\_API\_KEY='your-api-key-here'
```

4. 运行人工智能代理团队（AI Agents）

```
python xai\_finance\_agent.py
```

1. 打开网页浏览器，导航至控制台输出中提供的 URL，即可通过操作界面与人工智能财务代理进行交互。

### 补充说明

* **YFinance**：即 Yahoo Finance API 的简称，是一款常用的免费金融数据接口，可获取股票、基金等金融资产的实时及历史数据。

* **DuckDuckGo**：一款注重隐私保护的搜索引擎，其 API 可用于实现程序化网页搜索，避免用户数据被追踪。

* **环境变量设置**：上述`export`命令适用于 Linux/macOS 系统；若使用 Windows 系统，需在命令提示符（CMD）中执行`set XAI_API_KEY=your-api-key-here`，或在 PowerShell 中执行`$env:XAI_API_KEY="your-api-key-here"`。
