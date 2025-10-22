# OpenAI 研究员智能体（OpenAI Researcher Agent）

这是一款基于 OpenAI 智能体开发工具包（Agents SDK）和 Streamlit 构建的多智能体研究应用。用户可借助多个专业人工智能（AI）智能体，对任意主题开展全面研究。

### 功能特点（Features）

* **多智能体架构（Multi-Agent Architecture）**：

  * 分类智能体（Triage Agent）：规划研究方案并协调工作流程

  * 研究智能体（Research Agent）：浏览网络并收集相关信息

  * 编辑智能体（Editor Agent）：整合收集到的事实信息，形成完整报告

* **自动事实收集（Automatic Fact Collection）**：从研究中提取重要事实，并标注信息来源

* **结构化报告生成（Structured Report Generation）**：生成结构清晰的报告，包含标题、大纲及来源引用

* **交互式用户界面（Interactive UI）**：基于 Streamlit 构建，便于输入研究主题和查看结果

* **追踪与监控（Tracing and Monitoring）**：为整个研究流程集成追踪功能

### 如何开始使用？（How to get Started?）

1. 克隆 GitHub 仓库

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd awesome-llm-apps/ai\_agent\_tutorials/openai\_researcher\_agent
```

2. 安装所需依赖：

```
cd awesome-llm-apps/ai\_agent\_tutorials/openai\_researcher\_agent

pip install -r requirements.txt
```

3. 获取你的 OpenAI API 密钥

* 注册 [OpenAI 账号](https://platform.openai.com/) 并获取 API 密钥。

* 设置 OPENAI\_API\_KEY 环境变量：

```
export OPENAI\_API\_KEY='your-api-key-here'
```

4. 运行 AI 智能体团队

```
streamlit run openai\_researcher\_agent.py
```

随后打开浏览器，访问终端中显示的 URL（通常为 [http://localhost:8501](http://localhost:8501)）。

### 研究流程（Research Process）

1. 在侧边栏输入研究主题，或选择提供的示例主题之一

2. 点击 “开始研究（Start Research）” 启动流程

3. 在 “研究进程（Research Process）” 标签页实时查看研究过程

4. 研究完成后，切换至 “报告（Report）” 标签页，查看并下载生成的报告
