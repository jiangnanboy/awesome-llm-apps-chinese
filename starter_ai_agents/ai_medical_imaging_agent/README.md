# 🩻 医学影像诊断智能体（Medical Imaging Diagnosis Agent）

本医学影像诊断智能体基于 agno 平台构建，由 Gemini 2.0 Flash 提供技术支持，可对各类扫描产生的医学影像进行人工智能辅助分析。该智能体充当医学影像诊断专家的角色，能够分析多种类型的医学图像与视频，并提供详细的诊断见解及解释。

## 功能特点（Features）

* **全面的图像分析（Comprehensive Image Analysis）**

  * 图像类型识别（X 光片、磁共振成像 MRI、计算机断层扫描 CT、超声图像）

  * 解剖部位检测

  * 关键发现与观察结果记录

  * 潜在异常检测

  * 图像质量评估

  * 研究与参考资料支持

## 运行方法（How to Run）

1. **环境搭建（Setup Environment）**


```
\# 克隆代码仓库

git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git

cd ai\_agent\_tutorials/ai\_medical\_imaging\_agent

\# 安装依赖包

pip install -r requirements.txt
```

2. **配置 API 密钥（Configure API Keys）**

* 从[Google AI Studio（Google 人工智能工作室）](https://aistudio.google.com)获取 Google API 密钥

3. **运行应用程序（Run the Application）**

```
streamlit run ai\_medical\_imaging.py
```

## 分析组件（Analysis Components）

* **图像类型与部位（Image Type and Region）**

  * 识别成像模态（如 X 光、CT 等）

  * 明确解剖部位（如肺部、膝关节等）

* **关键发现（Key Findings）**

  * 系统性罗列观察结果

  * 详细描述影像表现

  * 突出异常区域

* **诊断评估（Diagnostic Assessment）**

  * 对潜在诊断结果进行排序

  * 提供鉴别诊断（区分相似病症）

  * 评估病情严重程度

* **患者友好型解释（Patient-Friendly Explanations）**

  * 使用简化术语（避免专业术语堆砌）

  * 基于基础原理进行详细解释

  * 提供影像参考点位（帮助患者理解关键区域）

## 注意事项（Notes）

* 采用 Gemini 2.0 Flash 进行分析

* 需稳定的互联网连接

* API 免费使用额度：Google 提供每日 1500 次免费请求

* 仅用于教育与开发目的

* 不能替代专业医疗诊断

## 免责声明（Disclaimer）

本工具仅用于教育与信息参考目的。所有分析结果均需由具备资质的医疗专业人员审核。请勿仅依据本工具的分析结果做出医疗决策。
