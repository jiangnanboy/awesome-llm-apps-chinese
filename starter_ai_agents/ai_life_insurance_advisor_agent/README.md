# 🛡️ 人寿保险保障顾问代理（Life Insurance Coverage Advisor Agent）

这是一款基于 Streamlit 框架开发的应用程序，可帮助用户估算其可能需要的定期人寿保险保额，并展示当前可用的保险产品选项。该应用由**Agno 代理框架**提供技术支持，采用**OpenAI GPT-5**作为大语言模型（LLM），借助**E2B 沙箱**实现确定性的保障额度计算，并通过**Firecrawl**进行实时网络检索。

## 主要亮点（Highlights）

* 极简信息填写表单（包含年龄、收入、受抚养人数量、债务、资产、现有保障额度、保障期限、所在地区）。

* 代理会在 E2B 沙箱内运行 Python 代码，通过贴现现金流式收入替代模型计算所需保障额度。

* 利用 Firecrawl 检索功能，根据用户所在地区及保障需求，收集最新的定期人寿保险产品信息。

* 输出简洁的保障额度估算结果、计算明细，以及最多三款保险产品建议（含信息来源链接）。

## 前置条件（Prerequisites）

使用该应用需获取各外部服务的 API 密钥，具体如下：

| 服务（Service）        | 用途（Purpose） | 获取地址（Where to get it）                                                            |
| ------------------ | ----------- | -------------------------------------------------------------------------------- |
| OpenAI（GPT-5-mini） | 核心推理模型      | [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)     |
| Firecrawl          | 网络检索与抓取工具   | [https://www.firecrawl.dev/app/api-keys](https://www.firecrawl.dev/app/api-keys) |
| E2B                | 安全代码执行沙箱    | [https://e2b.dev](https://e2b.dev)                                               |

## 安装步骤（Installation）

1. 克隆 GitHub 仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
```

2. 创建并激活虚拟环境（可选，但推荐）。

3. 安装依赖包：

```
pip install -r requirements.txt
```

4. 运行 Streamlit 应用：

```
streamlit run life\_insurance\_advisor\_agent.py
```

## 应用使用方法（Using the App）


1. 在侧边栏输入 OpenAI、Firecrawl 和 E2B 的 API 密钥（密钥仅存储在本地 Streamlit 会话中）。

2. 提供所需的财务信息，并选择收入替代保障期限。

3. 点击**生成保障额度与产品选项（Generate Coverage & Options）** ，启动 Agno 代理工作流程。

4. 查看推荐的保障额度、计算依据及建议的保险公司。可通过展开面板查看代理原始输出，以便进行调试。

## 免责声明（Disclaimer）

本项目仅用于教育和原型开发目的，**不提供持牌金融咨询服务**。请务必通过合格专业人士验证输出结果，并直接与保险公司确认相关细节。

