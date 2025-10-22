import requests
import os
import io
import base64
import PIL
from PIL import Image
import tqdm
import numpy as np
import streamlit as st
import cohere
from google import genai
import fitz  # PyMuPDF，用于处理PDF文件

# --- Streamlit应用配置 ---
st.set_page_config(layout="wide", page_title="基于Cohere Embed-4的视觉RAG")
st.title("视觉RAG with Cohere Embed-4 🖼️")

# --- API密钥输入区域 ---
with st.sidebar:
    st.header("🔑 API密钥")
    cohere_api_key = st.text_input("Cohere API密钥", type="password", key="cohere_key")
    google_api_key = st.text_input("Google API密钥 (Gemini)", type="password", key="google_key")
    "[获取Cohere API密钥](https://dashboard.cohere.com/api-keys)"
    "[获取Google API密钥](https://aistudio.google.com/app/apikey)"

    st.markdown("---")
    # 检查密钥是否输入
    if not cohere_api_key:
        st.warning("请输入Cohere API密钥以继续")
    if not google_api_key:
        st.warning("请输入Google API密钥以继续")
    st.markdown("---")

# --- 初始化API客户端 ---
co = None  # Cohere客户端实例
genai_client = None  # Google Gemini客户端实例
# 初始化会话状态用于存储嵌入和图像路径
if 'image_paths' not in st.session_state:
    st.session_state.image_paths = []  # 存储所有处理过的图像路径
if 'doc_embeddings' not in st.session_state:
    st.session_state.doc_embeddings = None  # 存储所有图像的嵌入向量

# 当两个API密钥都提供时，初始化客户端
if cohere_api_key and google_api_key:
    try:
        co = cohere.ClientV2(api_key=cohere_api_key)
        st.sidebar.success("Cohere客户端初始化成功!")
    except Exception as e:
        st.sidebar.error(f"Cohere初始化失败: {e}")

    try:
        genai_client = genai.Client(api_key=google_api_key)
        st.sidebar.success("Gemini客户端初始化成功!")
    except Exception as e:
        st.sidebar.error(f"Gemini初始化失败: {e}")
else:
    st.info("请在侧边栏输入API密钥以开始使用")

# 模型信息说明
with st.expander("ℹ️ 使用的模型说明"):
    st.markdown("""
    ### Cohere Embed-4

    Cohere的Embed-4是最先进的多模态嵌入模型，专为企业搜索和检索设计。
    它支持：

    - **多模态搜索**：无缝融合文本和图像搜索
    - **高精度**：在检索任务中表现领先
    - **高效嵌入**：处理复杂图像（如图表、图形和信息图）

    该模型无需复杂的OCR预处理即可处理图像，并保持视觉元素与文本之间的关联。

    ### Google Gemini 2.5 Flash

    Gemini 2.5 Flash是谷歌的高效多模态模型，可处理文本和图像输入以生成高质量响应。
    它专为快速推理而设计，同时保持高精度，非常适合此类RAG系统的实时应用。
    """)

# --- 辅助函数 ---
# 一些用于调整图像大小和转换为base64格式的辅助函数
max_pixels = 1568 * 1568  # 图像的最大分辨率限制


# 调整过大的图像
def resize_image(pil_image: PIL.Image.Image) -> None:
    """如果图像超过最大像素限制，则就地调整大小"""
    org_width, org_height = pil_image.size

    # 如果图像过大则调整大小
    if org_width * org_height > max_pixels:
        # 计算缩放因子以确保总像素不超过限制
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))


# 将图像转换为base64字符串后发送到API
def base64_from_image(img_path: str) -> str:
    """将图像文件转换为base64编码的字符串"""
    pil_image = PIL.Image.open(img_path)
    img_format = pil_image.format if pil_image.format else "PNG"  # 默认使用PNG格式

    resize_image(pil_image)  # 确保图像大小符合要求

    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        # 生成base64编码的图像数据（包含数据类型前缀）
        img_data = f"data:image/{img_format.lower()};base64," + base64.b64encode(img_buffer.read()).decode("utf-8")

    return img_data


# 将PIL图像转换为base64字符串
def pil_to_base64(pil_image: PIL.Image.Image) -> str:
    """将PIL图像对象转换为base64编码的字符串"""
    if pil_image.format is None:
        img_format = "PNG"  # 默认为PNG格式
    else:
        img_format = pil_image.format

    resize_image(pil_image)  # 调整图像大小

    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64," + base64.b64encode(img_buffer.read()).decode("utf-8")

    return img_data


# 计算图像的嵌入向量
@st.cache_data(ttl=3600, show_spinner=False)
def compute_image_embedding(base64_img: str, _cohere_client) -> np.ndarray | None:
    """使用Cohere的Embed-4模型计算图像的嵌入向量"""
    try:
        api_response = _cohere_client.embed(
            model="embed-v4.0",  # 使用最新的嵌入模型
            input_type="search_document",  # 表示这是文档嵌入（用于检索）
            embedding_types=["float"],  # 请求浮点类型的嵌入
            images=[base64_img],  # 传入base64编码的图像
        )

        # 提取嵌入向量
        if api_response.embeddings and api_response.embeddings.float:
            return np.asarray(api_response.embeddings.float[0])
        else:
            st.warning("无法获取嵌入向量，API响应可能为空")
            return None
    except Exception as e:
        st.error(f"计算嵌入向量时出错: {e}")
        return None


# 处理PDF文件：提取页面作为图像并生成嵌入
def process_pdf_file(pdf_file, cohere_client, base_output_folder="pdf_pages") -> tuple[
    list[str], list[np.ndarray] | None]:
    """从PDF中提取页面作为图像，生成嵌入并保存

    参数:
        pdf_file: Streamlit的上传文件对象
        cohere_client: 初始化的Cohere客户端
        base_output_folder: 保存页面图像的目录

    返回:
        包含以下内容的元组:
          - 保存的页面图像路径列表
          - 每个页面的numpy数组嵌入列表，若嵌入失败则为None
    """
    page_image_paths = []  # 存储页面图像路径
    page_embeddings = []  # 存储页面嵌入向量
    pdf_filename = pdf_file.name
    # 创建输出目录（按PDF文件名分类）
    output_folder = os.path.join(base_output_folder, os.path.splitext(pdf_filename)[0])
    os.makedirs(output_folder, exist_ok=True)

    try:
        # 从流中打开PDF
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        st.write(f"处理PDF: {pdf_filename}（共{len(doc)}页）")
        pdf_progress = st.progress(0.0)  # 显示处理进度

        for i, page in enumerate(doc.pages()):
            page_num = i + 1
            page_img_path = os.path.join(output_folder, f"page_{page_num}.png")
            page_image_paths.append(page_img_path)

            # 将页面渲染为图像
            pix = page.get_pixmap(dpi=150)  # 调整DPI平衡质量和性能
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # 临时保存页面图像
            pil_image.save(page_img_path, "PNG")

            # 将PIL图像转换为base64
            base64_img = pil_to_base64(pil_image)

            # 计算页面图像的嵌入
            emb = compute_image_embedding(base64_img, _cohere_client=cohere_client)
            if emb is not None:
                page_embeddings.append(emb)
            else:
                st.warning(f"无法为{pdf_filename}的第{page_num}页生成嵌入，已跳过")
                page_embeddings.append(None)  # 添加占位符保持列表对齐

            # 更新进度条
            pdf_progress.progress((i + 1) / len(doc))

        doc.close()
        pdf_progress.empty()  # 完成后移除进度条

        # 过滤掉嵌入失败的页面
        valid_paths = [path for i, path in enumerate(page_image_paths) if page_embeddings[i] is not None]
        valid_embeddings = [emb for emb in page_embeddings if emb is not None]

        if not valid_embeddings:
            st.error(f"无法为{pdf_filename}生成任何有效嵌入")
            return [], None

        return valid_paths, valid_embeddings

    except Exception as e:
        st.error(f"处理PDF {pdf_filename}时出错: {e}")
        return [], None


# 下载并嵌入样本图像
@st.cache_data(ttl=3600, show_spinner=False)
def download_and_embed_sample_images(_cohere_client) -> tuple[list[str], np.ndarray | None]:
    """下载样本图像并使用Cohere的Embed-4模型计算其嵌入向量"""
    # 来自https://www.appeconomyinsights.com/的几张图像
    images = {
        "tesla.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbef936e6-3efa-43b3-88d7-7ec620cdb33b_2744x1539.png",
        "netflix.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F23bd84c9-5b62-4526-b467-3088e27e4193_2744x1539.png",
        "nike.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5cd33ba-ae1a-42a8-a254-d85e690d9870_2741x1541.png",
        "google.png": "https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F395dd3b9-b38e-4d1f-91bc-d37b642ee920_2741x1541.png",
        "accenture.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08b2227c-7dc8-49f7-b3c5-13cab5443ba6_2741x1541.png",
        "tecent.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0ec8448c-c4d1-4aab-a8e9-2ddebe0c95fd_2741x1541.png"
    }

    # 准备文件夹
    img_folder = "img"
    os.makedirs(img_folder, exist_ok=True)

    img_paths = []  # 存储图像路径
    doc_embeddings = []  # 存储嵌入向量

    # 使用st.spinner包装TQDM以获得更好的UI体验
    with st.spinner("正在下载并嵌入样本图像..."):
        pbar = tqdm.tqdm(images.items(), desc="处理样本图像")
        for name, url in pbar:
            img_path = os.path.join(img_folder, name)
            # 如果尚未处理，则添加路径
            if img_path not in img_paths:
                img_paths.append(img_path)

                # 下载图像（如果不存在）
                if not os.path.exists(img_path):
                    try:
                        response = requests.get(url)
                        response.raise_for_status()  # 检查请求是否成功
                        with open(img_path, "wb") as fOut:
                            fOut.write(response.content)
                    except requests.exceptions.RequestException as e:
                        st.error(f"下载{name}失败: {e}")
                        img_paths.pop()  # 移除失败的路径
                        continue

            # 计算图像嵌入（如果尚未计算）
            current_index = -1
            try:
                current_index = img_paths.index(img_path)
            except ValueError:
                continue  # 路径不在列表中，跳过

            # 检查该索引是否已有嵌入
            if current_index >= len(doc_embeddings):
                try:
                    # 确保文件存在后再计算嵌入
                    if os.path.exists(img_path):
                        base64_img = base64_from_image(img_path)
                        emb = compute_image_embedding(base64_img, _cohere_client=_cohere_client)
                        if emb is not None:
                            # 确保嵌入列表长度与路径列表一致
                            while len(doc_embeddings) < current_index:
                                doc_embeddings.append(None)
                            doc_embeddings.append(emb)
                    else:
                        # 文件不存在时添加占位符
                        while len(doc_embeddings) < current_index:
                            doc_embeddings.append(None)
                        doc_embeddings.append(None)
                except Exception as e:
                    st.error(f"嵌入{name}失败: {e}")
                    while len(doc_embeddings) < current_index:
                        doc_embeddings.append(None)
                    doc_embeddings.append(None)

    # 过滤无效嵌入及其对应的路径
    filtered_paths = [path for i, path in enumerate(img_paths) if
                      i < len(doc_embeddings) and doc_embeddings[i] is not None]
    filtered_embeddings = [emb for emb in doc_embeddings if emb is not None]

    if filtered_embeddings:
        doc_embeddings_array = np.vstack(filtered_embeddings)  # 堆叠为二维数组
        return filtered_paths, doc_embeddings_array

    return [], None


# 搜索函数
def search(question: str, co_client: cohere.Client, embeddings: np.ndarray, image_paths: list[str],
           max_img_size: int = 800) -> str | None:
    """为给定问题找到最相关的图像路径"""
    # 检查必要条件是否满足
    if not co_client or embeddings is None or embeddings.size == 0 or not image_paths:
        st.warning("搜索前置条件不满足（客户端、嵌入或路径缺失/为空）")
        return None
    if embeddings.shape[0] != len(image_paths):
        st.error(f"嵌入数量({embeddings.shape[0]})与图像路径数量({len(image_paths)})不匹配，无法执行搜索")
        return None

    try:
        # 计算查询的嵌入向量
        api_response = co_client.embed(
            model="embed-v4.0",
            input_type="search_query",  # 表示这是查询嵌入
            embedding_types=["float"],
            texts=[question],  # 传入问题文本
        )

        if not api_response.embeddings or not api_response.embeddings.float:
            st.error("无法获取查询嵌入")
            return None

        query_emb = np.asarray(api_response.embeddings.float[0])

        # 确保查询嵌入维度与文档嵌入一致
        if query_emb.shape[0] != embeddings.shape[1]:
            st.error(f"查询嵌入维度({query_emb.shape[0]})与文档嵌入维度({embeddings.shape[1]})不匹配")
            return None

        # 计算余弦相似度（点积，因为嵌入已归一化）
        cos_sim_scores = np.dot(query_emb, embeddings.T)

        # 获取最相关的图像
        top_idx = np.argmax(cos_sim_scores)
        hit_img_path = image_paths[top_idx]
        print(f"问题: {question}")  # 调试用
        print(f"最相关图像: {hit_img_path}")  # 调试用

        return hit_img_path
    except Exception as e:
        st.error(f"搜索过程中出错: {e}")
        return None


# 回答函数
def answer(question: str, img_path: str, gemini_client) -> str:
    """使用Gemini根据提供的图像回答问题"""
    # 检查必要条件
    if not gemini_client or not img_path or not os.path.exists(img_path):
        missing = []
        if not gemini_client: missing.append("Gemini客户端")
        if not img_path:
            missing.append("图像路径")
        elif not os.path.exists(img_path):
            missing.append(f"图像文件 {img_path}")
        return f"回答前置条件不满足（{', '.join(missing)}缺失或无效）"
    try:
        img = PIL.Image.open(img_path)
        prompt = [f"""根据以下图像回答问题。尽可能详细地提供相关信息。
不要在回答中使用markdown格式。
请提供足够的上下文。

问题: {question}""", img]  # 提示词+图像

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",  # 使用Gemini 2.5 Flash模型
            contents=prompt
        )

        llm_answer = response.text
        print("LLM回答:", llm_answer)  # 调试用
        return llm_answer
    except Exception as e:
        st.error(f"生成回答时出错: {e}")
        return f"生成回答失败: {e}"


# --- 主UI设置 ---
st.subheader("📊 加载样本图像")
if cohere_api_key and co:
    # 点击按钮时加载样本图像到会话状态
    if st.button("加载样本图像", key="load_sample_button"):
        sample_img_paths, sample_doc_embeddings = download_and_embed_sample_images(_cohere_client=co)
        if sample_img_paths and sample_doc_embeddings is not None:
            # 将样本图像添加到会话状态（避免重复）
            current_paths = set(st.session_state.image_paths)
            new_paths = [p for p in sample_img_paths if p not in current_paths]

            if new_paths:
                # 提取新图像对应的嵌入
                new_embeddings_to_add = sample_doc_embeddings[
                    [idx for idx, p in enumerate(sample_img_paths) if p in new_paths]]

                # 更新会话状态
                st.session_state.image_paths.extend(new_paths)
                if st.session_state.doc_embeddings is None or st.session_state.doc_embeddings.size == 0:
                    st.session_state.doc_embeddings = new_embeddings_to_add
                else:
                    st.session_state.doc_embeddings = np.vstack(
                        (st.session_state.doc_embeddings, new_embeddings_to_add))
                st.success(f"已加载{len(new_paths)}张样本图像")
            else:
                st.info("样本图像已加载")
        else:
            st.error("加载样本图像失败，请查看控制台错误信息")
else:
    st.warning("输入API密钥以启用样本图像加载")

st.markdown("--- ")
# --- 文件上传区域（主UI） ---
st.subheader("📤 上传你的图像")
st.info("或者，上传你自己的图像或PDF。RAG过程将搜索所有已加载的内容。")

# 文件上传器
uploaded_files = st.file_uploader("上传图像（PNG、JPG、JPEG）或PDF",
                                  type=["png", "jpg", "jpeg", "pdf"],
                                  accept_multiple_files=True, key="image_uploader",
                                  label_visibility="collapsed")

# 处理上传的图像
if uploaded_files and co:
    st.write(f"正在处理{len(uploaded_files)}个上传的文件...")
    progress_bar = st.progress(0)

    # 创建上传图像的临时目录
    upload_folder = "uploaded_img"
    os.makedirs(upload_folder, exist_ok=True)

    newly_uploaded_paths = []  # 新上传的图像路径
    newly_uploaded_embeddings = []  # 新上传图像的嵌入

    for i, uploaded_file in enumerate(uploaded_files):
        # 检查是否已处理过该文件（简单的文件名检查）
        img_path = os.path.join(upload_folder, uploaded_file.name)
        if img_path not in st.session_state.image_paths:
            try:
                # 检查文件类型
                file_type = uploaded_file.type
                if file_type == "application/pdf":
                    # 处理PDF - 返回页面路径列表和嵌入列表
                    pdf_page_paths, pdf_page_embeddings = process_pdf_file(uploaded_file, cohere_client=co)
                    if pdf_page_paths and pdf_page_embeddings:
                        # 只添加不在会话状态中的路径/嵌入
                        current_paths_set = set(st.session_state.image_paths)
                        unique_new_paths = [p for p in pdf_page_paths if p not in current_paths_set]
                        if unique_new_paths:
                            indices_to_add = [i for i, p in enumerate(pdf_page_paths) if p in unique_new_paths]
                            newly_uploaded_paths.extend(unique_new_paths)
                            newly_uploaded_embeddings.extend([pdf_page_embeddings[idx] for idx in indices_to_add])
                elif file_type in ["image/png", "image/jpeg"]:
                    # 处理常规图像
                    # 保存上传的文件
                    with open(img_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # 获取嵌入
                    base64_img = base64_from_image(img_path)
                    emb = compute_image_embedding(base64_img, _cohere_client=co)

                    if emb is not None:
                        newly_uploaded_paths.append(img_path)
                        newly_uploaded_embeddings.append(emb)
                else:
                    st.warning(f"跳过不支持的文件类型: {uploaded_file.name}（{file_type}）")

            except Exception as e:
                st.error(f"处理{uploaded_file.name}时出错: {e}")
        # 更新进度条（无论处理状态如何，提供用户反馈）
        progress_bar.progress((i + 1) / len(uploaded_files))

    # 将新处理的文件添加到会话状态
    if newly_uploaded_paths:
        st.session_state.image_paths.extend(newly_uploaded_paths)
        if newly_uploaded_embeddings:
            new_embeddings_array = np.vstack(newly_uploaded_embeddings)
            if st.session_state.doc_embeddings is None or st.session_state.doc_embeddings.size == 0:
                st.session_state.doc_embeddings = new_embeddings_array
            else:
                st.session_state.doc_embeddings = np.vstack((st.session_state.doc_embeddings, new_embeddings_array))
            st.success(f"成功处理并添加了{len(newly_uploaded_paths)}张新图像")
        else:
            st.warning("无法为新上传的图像生成嵌入")
    elif uploaded_files:  # 选择了文件但没有新文件
        st.info("所选图像似乎已处理过")

# --- 视觉RAG区域（主UI） ---
st.markdown("---")
st.subheader("❓ 提问")

if not st.session_state.image_paths:
    st.warning("请先加载样本图像或上传你自己的图像")
else:
    st.info(f"已准备好回答关于{len(st.session_state.image_paths)}张图像的问题")

    # 显示所有已加载图像的缩略图（可选）
    with st.expander("查看已加载的图像", expanded=False):
        if st.session_state.image_paths:
            num_images_to_show = len(st.session_state.image_paths)
            cols = st.columns(5)  # 每行显示5个缩略图
            for i in range(num_images_to_show):
                with cols[i % 5]:
                    # 显示时处理可能的文件缺失
                    try:
                        st.image(st.session_state.image_paths[i], width=100,
                                 caption=os.path.basename(st.session_state.image_paths[i]))
                    except FileNotFoundError:
                        st.error(f"缺失文件: {os.path.basename(st.session_state.image_paths[i])}")
        else:
            st.write("尚未加载图像")

# 问题输入框
question = st.text_input("请输入关于已加载图像的问题:",
                         key="main_question_input",
                         placeholder="例如：耐克的净利润是多少？",
                         disabled=not st.session_state.image_paths)

# 运行按钮（只有满足所有条件才启用）
run_button = st.button("运行视觉RAG", key="main_run_button",
                       disabled=not (
                                   cohere_api_key and google_api_key and question and st.session_state.image_paths and st.session_state.doc_embeddings is not None and st.session_state.doc_embeddings.size > 0))

# 输出区域
st.markdown("### 结果")
retrieved_image_placeholder = st.empty()  # 用于显示检索到的图像
answer_placeholder = st.empty()  # 用于显示回答

# 运行搜索和回答逻辑
if run_button:
    if co and genai_client and st.session_state.doc_embeddings is not None and len(st.session_state.doc_embeddings) > 0:
        with st.spinner("正在查找相关图像..."):
            # 确保嵌入和路径数量匹配
            if len(st.session_state.image_paths) != st.session_state.doc_embeddings.shape[0]:
                st.error("错误：图像数量与嵌入数量不匹配，无法继续")
            else:
                top_image_path = search(question, co, st.session_state.doc_embeddings, st.session_state.image_paths)

                if top_image_path:
                    # 生成图像标题
                    caption = f"为问题检索到的内容: '{question}'（来源: {os.path.basename(top_image_path)}）"
                    # 如果是PDF页面图像，显示更详细的来源信息
                    if top_image_path.startswith("pdf_pages/"):
                        parts = top_image_path.split(os.sep)
                        if len(parts) >= 3:
                            pdf_name = parts[1]
                            page_name = parts[-1]
                            caption = f"为问题检索到的内容: '{question}'（来源: {pdf_name}.pdf, {page_name.replace('.png', '')}）"

                    retrieved_image_placeholder.image(top_image_path, caption=caption, use_container_width=True)

                    with st.spinner("正在生成回答..."):
                        final_answer = answer(question, top_image_path, genai_client)
                        answer_placeholder.markdown(f"**回答:**\n{final_answer}")
                else:
                    retrieved_image_placeholder.warning("无法找到与您的问题相关的图像")
                    answer_placeholder.text("")  # 清空回答区域
    else:
        # 此情况理论上会被按钮的disabled状态阻止
        st.error("无法运行RAG。请检查API客户端并确保图像已加载且生成了嵌入")

# 页脚
st.markdown("---")
st.caption("基于Cohere Embed-4的视觉RAG | 使用Streamlit、Cohere Embed-4和Google Gemini 2.5 Flash构建")