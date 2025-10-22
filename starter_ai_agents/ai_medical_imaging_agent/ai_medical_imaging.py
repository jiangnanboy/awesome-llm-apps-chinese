import os
from PIL import Image as PILImage  # 导入PIL库的Image模块，用于图像处理，并重命名为PILImage以避免命名冲突
from agno.agent import Agent  # 导入agno框架的Agent类，用于创建智能体
from agno.models.google import Gemini  # 导入agno框架中对接Google Gemini模型的类
import streamlit as st  # 导入Streamlit库，用于构建Web应用界面，简写为st
from agno.tools.duckduckgo import DuckDuckGoTools  # 导入agno框架的DuckDuckGo搜索工具
from agno.media import Image as AgnoImage  # 导入agno框架的Image类，用于处理媒体图像，重命名为AgnoImage

# 检查Streamlit会话状态中是否存在"GOOGLE_API_KEY"（Google API密钥），若不存在则初始化为None
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None

# 在Streamlit应用的侧边栏中添加内容
with st.sidebar:
    st.title("ℹ️ 配置（Configuration）")  # 侧边栏标题

    # 若会话状态中无API密钥，则显示输入框让用户填写
    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input(
            "请输入您的Google API密钥：",  # 输入框提示文本
            type="password"  # 输入类型为密码，输入内容会隐藏
        )
        # 显示获取API密钥的提示及链接
        st.caption(
            "请从 [Google AI Studio]（https://aistudio.google.com/apikey）获取您的API密钥 🔑"
        )
        # 若用户输入了API密钥，则保存到会话状态并刷新页面
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("API密钥已保存！")  # 显示成功提示
            st.rerun()  # 重新运行应用以应用配置
    # 若会话状态中已存在API密钥，则显示配置成功信息
    else:
        st.success("API密钥已配置")
        # 显示"重置API密钥"按钮，点击后清除会话状态中的密钥并刷新页面
        if st.button("🔄 重置API密钥"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()

    # 显示工具功能说明信息
    st.info(
        "本工具结合先进的计算机视觉技术和放射学专业知识，为医学影像数据提供AI驱动的分析服务。"
    )
    # 显示免责声明警告
    st.warning(
        "⚠ 免责声明（DISCLAIMER）：本工具仅用于教育和信息参考目的。"
        "所有分析结果均需由具备资质的医疗专业人员审核。"
        "请勿仅凭本分析结果做出医疗决策。"
    )

# 若会话状态中存在API密钥，则创建医疗分析智能体；否则设为None
medical_agent = Agent(
    model=Gemini(  # 使用Google Gemini模型
        id="gemini-2.0-flash",  # 模型ID，指定使用Gemini 2.0 Flash版本
        api_key=st.session_state.GOOGLE_API_KEY  # 传入用户配置的API密钥
    ),
    tools=[DuckDuckGoTools()],  # 为智能体配置DuckDuckGo搜索工具
    markdown=True  # 启用Markdown格式输出
) if st.session_state.GOOGLE_API_KEY else None

# 若智能体未创建（即API密钥未配置），则显示警告提示
if not medical_agent:
    st.warning("请在侧边栏配置您的API密钥以继续使用")

# 医疗分析指令（定义智能体的分析任务和输出格式）
query = """
你是一位具备丰富放射学和诊断影像知识的高技能医学影像专家。请分析患者的医学影像，并按以下结构组织回复：

### 1. 影像类型与部位（Image Type & Region）
- 明确影像检查方式（X线/磁共振成像MRI/计算机断层扫描CT/超声等）
- 确定患者的解剖部位及体位
- 评价影像质量和技术适用性

### 2. 主要发现（Key Findings）
- 系统列出主要观察结果
- 详细描述影像中发现的任何异常情况
- 相关处注明测量数据和密度信息
- 说明异常的位置、大小、形态及特征
- 评估严重程度：正常/轻度/中度/重度

### 3. 诊断评估（Diagnostic Assessment）
- 给出主要诊断结果及置信度
- 按可能性顺序列出鉴别诊断
- 为每个诊断结果提供影像中观察到的支持证据
- 注明任何危急或紧急发现

### 4. 患者易懂解释（Patient-Friendly Explanation）
- 用患者可理解的简单、清晰语言解释检查结果
- 避免使用专业医学术语，若必须使用需提供明确释义
- 必要时使用直观类比辅助理解
- 解答患者针对此类结果可能存在的常见疑问

### 5. 研究背景（Research Context）
重要提示（IMPORTANT）：请使用DuckDuckGo搜索工具完成以下任务：
- 查找与类似病例相关的最新医学文献
- 搜索相关疾病的标准治疗方案
- 提供相关医学链接列表
- 研究该领域的相关技术进展
- 引用2-3篇关键参考文献以支持分析结论

请使用清晰的Markdown标题和项目符号格式化回复，要求简洁且内容全面。
"""

# 应用主页面内容
st.title("🏥 医学影像诊断智能体（Medical Imaging Diagnosis Agent）")
st.write("上传医学影像以获取专业分析")

# 创建容器用于更好地组织页面布局
upload_container = st.container()  # 影像上传容器
image_container = st.container()  # 影像显示容器
analysis_container = st.container()  # 分析结果容器

# 在影像上传容器中添加文件上传组件
with upload_container:
    uploaded_file = st.file_uploader(
        "上传医学影像",  # 上传组件提示文本
        type=["jpg", "jpeg", "png", "dicom"],  # 支持的文件格式
        help="支持格式：JPG、JPEG、PNG、DICOM"  # 帮助提示信息
    )

# 若用户已上传文件，则执行以下操作
if uploaded_file is not None:
    with image_container:
        # 创建3列布局，中间列用于显示影像（左右列留白以居中）
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # 用PIL打开上传的影像文件
            image = PILImage.open(uploaded_file)
            # 获取原始影像的宽高和宽高比
            width, height = image.size
            aspect_ratio = width / height
            # 设置新的宽度并按原比例计算新高度，保证影像不失真
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            # 调整影像大小
            resized_image = image.resize((new_width, new_height))

            # 在页面中显示调整后的影像
            st.image(
                resized_image,
                caption="已上传的医学影像",  # 影像标题
                use_container_width=True  # 启用容器宽度自适应
            )

            # 添加"分析影像"按钮（-primary表示强调样式，use_container_width表示按钮宽度自适应）
            analyze_button = st.button(
                "🔍 分析影像",
                type="primary",
                use_container_width=True
            )

    # 在分析结果容器中处理分析逻辑
    with analysis_container:
        # 若用户点击了"分析影像"按钮
        if analyze_button:
            # 显示加载中动画和提示文本
            with st.spinner("🔄 正在分析影像... 请稍候。"):
                try:
                    # 定义临时文件路径，用于保存调整后的影像
                    temp_path = "temp_resized_image.png"
                    # 保存调整后的影像到临时路径
                    resized_image.save(temp_path)

                    # 创建agno框架的Image对象（用于智能体处理影像，若构造函数参数不同需调整）
                    agno_image = AgnoImage(filepath=temp_path)

                    # 调用智能体执行分析（传入指令和影像）
                    response = medical_agent.run(query, images=[agno_image])
                    # 显示分析结果标题
                    st.markdown("### 📋 分析结果")
                    st.markdown("---")  # 添加分隔线
                    # 显示智能体返回的分析内容（Markdown格式）
                    st.markdown(response.content)
                    st.markdown("---")  # 添加分隔线
                    # 显示结果备注信息
                    st.caption(
                        "注：本分析结果由AI生成，需由具备资质的医疗专业人员审核。"
                    )
                # 若分析过程中出现异常，显示错误信息
                except Exception as e:
                    st.error(f"分析出错：{e}")
# 若用户未上传文件，显示提示信息
else:
    st.info("👆 请上传医学影像以开始分析")