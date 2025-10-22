# Contextual AI RAG Agent：基于托管 RAG 平台的智能检索增强生成应用

这是一款集成了 Contextual AI 托管 RAG（检索增强生成）平台的 Streamlit 应用。通过该应用，用户可完成数据存储库创建、文档摄入、智能体启动等操作，并基于自有数据与智能体进行对话，确保对话内容始终以数据为依据，提升回答的准确性与可信度。

## 核心功能

* **文档摄入至数据存储库**：支持将各类文档（PDF、文本文件等）上传并摄入到 Contextual AI 的数据存储库中，为后续检索提供数据基础。

* **绑定数据存储库的智能体创建**：可创建智能体，并将其与一个或多个数据存储库进行绑定，确保智能体仅基于绑定存储库中的数据生成回答。

* **基于 Grounder 语言模型（GLM）的回答生成**：借助 Contextual 的 Grounder 语言模型，生成忠实于检索结果、以检索数据为支撑的回答，避免无依据的虚构内容。

* **多语言文档重排序**：根据查询相关性与自定义指令（支持多语言），对检索到的文档进行重排序，优先呈现与查询最匹配的内容。

* **检索可视化展示**：直观展示检索内容的来源信息，包括文档属性页面图片及元数据（如文档名称、页码、上传时间等），增强回答的可追溯性。

* **基于 LMUnit 的回答评估**：支持通过自定义评估标准，利用 LMUnit 工具对生成的回答进行评估，检验回答是否符合预期要求。

## 前置条件

* 拥有 Contextual AI 账号及对应的 API 密钥（获取路径：Contextual AI 控制台 → “API Keys” 菜单）。

### API 密钥生成步骤

1. 访问`app.contextual.ai`，登录你的租户账号。

2. 在控制台界面中，点击 “API Keys” 选项。

3. 点击 “Create API Key” 按钮，生成新的 API 密钥。

4. 复制生成的 API 密钥，在应用侧边栏提示时粘贴使用。

## 应用运行步骤

1. **克隆仓库并进入应用目录**：

   通过 Git 命令克隆代码仓库，并导航至该应用对应的文件夹，具体命令如下：

```
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd awesome-llm-apps/rag\_tutorials/contextualai\_rag\_agent
```
2. **创建并激活虚拟环境**：

   为避免依赖冲突，建议创建独立的虚拟环境。以 Python 自带的 venv 工具为例，操作命令如下（不同操作系统命令略有差异，此处以 Windows 系统为例）：

```
\# 创建虚拟环境（环境名称可自定义，此处以contextualai-env为例）

python -m venv contextualai-env

\# 激活虚拟环境

contextualai-env\Scripts\activate
```

若为 macOS 或 Linux 系统，激活命令为：

```
source contextualai-env/bin/activate
```

3. **安装依赖包**：

   使用 pip 命令安装应用运行所需的全部依赖，依赖清单已包含在`requirements.txt`文件中，命令如下：

```
pip install -r requirements.txt
```

4. **启动应用**：

   通过 Streamlit 命令启动应用，具体命令如下：

```
streamlit run contextualai\_rag\_agent.py
```

执行命令后，系统会自动在默认浏览器中打开应用界面；若未自动打开，可在终端中找到类似 “Local URL: [http://localhost:8501](http://localhost:8501)” 的链接，复制到浏览器地址栏中访问。

## 使用流程

1. **API 密钥配置**：

   在应用侧边栏中，找到 “Contextual AI API Key” 输入框，粘贴之前生成的 API 密钥。若已拥有现成的智能体 ID（Agent ID）或数据存储库 ID（Datastore ID），可在对应输入框中填写，直接关联已有资源；若没有，则后续步骤中创建即可。

2. **数据存储库创建与文档摄入**：

   若无需使用已有数据存储库，可点击 “Create New Datastore” 按钮创建新的存储库。创建完成后，点击 “Upload Documents” 按钮，选择本地的 PDF 或文本文件进行上传。上传后，应用会自动处理文档并将其摄入到数据存储库中，需等待文档处理完成（处理进度会在界面中提示）。

3. **智能体创建与绑定**：

   点击 “Create New Agent” 按钮，在创建界面中选择需要绑定的数据存储库（可绑定一个或多个），完成智能体创建。若已有现成的智能体，可直接在侧边栏填写 Agent ID，应用会自动关联该智能体。

4. **与智能体对话**：

   在应用主界面的聊天输入框中，输入你的问题并发送。智能体会基于绑定数据存储库中的数据生成回答，并在回答下方展示检索到的相关文档信息（包括属性页面图片及元数据）。

5. **高级功能使用（可选）**：

* **智能体设置**：通过界面中的 “Agent Settings” 选项，可直接在 UI 界面中更新智能体的系统提示词，调整智能体的对话风格与回答逻辑。

* **调试与评估**：开启 “Retrieval Info” 开关，可查看更详细的检索归因信息；点击 “Run LMUnit Evaluation” 按钮，可针对上一次生成的回答，输入自定义评估标准（如 “回答是否准确覆盖文档核心内容”“语言表达是否流畅” 等），利用 LMUnit 工具对回答进行评分与评估。

## 配置说明

* **非美国云实例 Base URL 设置**：若你的 Contextual AI 账号使用的是非美国地区的云实例，需在应用侧边栏的 “Base URL” 输入框中填写对应的基础 URL（例如`http://api.contextual.ai/v1`）。应用会将该 URL 作为所有 API 调用（包括就绪性轮询）的基础地址，确保 API 请求能正确路由到对应的实例。

* **检索可视化配置**：检索可视化功能通过调用`agents.query.retrieval_info`接口获取文档页面的 base64 格式图片数据，并在界面中直接渲染展示，无需额外配置，确保用户能直观看到回答对应的文档来源页面。

* **LMUnit 评估配置**：LMUnit 评估功能通过调用`lmunit.create`接口，将用户自定义的评估标准与上一次的回答内容发送至 Contextual AI 平台，由平台完成评分计算，应用仅负责结果展示，无需用户进行额外的底层配置。
