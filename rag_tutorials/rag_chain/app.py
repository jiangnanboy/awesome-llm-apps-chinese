import os
import streamlit as st

from langchain_google_genai import GoogleGenererativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 初始化嵌入模型
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 初始化制药数据库
db = Chroma(collection_name="pharma_database",
            embedding_function=embedding_model,
            persist_directory='./pharma_db')


def format_docs(docs):
    """将文档对象列表格式化为单个字符串。

    参数:
        docs (list): 文档对象列表，每个对象都有'page_content'属性。

    返回:
        str: 包含每个文档页面内容的单个字符串，内容之间用双换行分隔。"""
    return "\n\n".join(doc.page_content for doc in docs)


def add_to_db(uploaded_files):
    """处理上传的PDF文件并添加到数据库。

    此函数检查是否有文件上传。如果有文件上传，
    将每个文件保存到临时位置，使用PDF加载器处理内容，
    并将内容分割成更小的块。每个块及其元数据随后被添加到数据库。
    处理后删除临时文件。

    参数:
        uploaded_files (list): 要处理的上传文件对象列表。

    返回:
        None"""
    # 检查是否有文件上传
    if not uploaded_files:
        st.error("未上传文件！")
        return

    for uploaded_file in uploaded_files:
        # 将上传的文件保存到临时路径
        temp_file_path = os.path.join("./temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        # 使用PyPDFLoader加载文件
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()

        # 存储元数据和内容
        doc_metadata = [data[i].metadata for i in range(len(data))]
        doc_content = [data[i].page_content for i in range(len(data))]

        # 将文档分割成更小的块
        st_text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",
            chunk_size=100,
            chunk_overlap=50
        )
        st_chunks = st_text_splitter.create_documents(doc_content, doc_metadata)

        # 将块添加到数据库
        db.add_documents(st_chunks)

        # 处理后删除临时文件
        os.remove(temp_file_path)


def run_rag_chain(query):
    """使用检索增强生成(RAG)链处理查询。

    此函数利用RAG链回答给定的查询。它使用相似性搜索检索相关上下文，
    然后使用专门研究制药科学的聊天模型基于此上下文生成响应。

    参数:
        query (str): 需要回答的用户问题。

    返回:
        str: 聊天模型基于检索到的上下文生成的响应。"""
    # 创建检索器对象并应用相似性搜索
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    # 初始化聊天提示模板
    PROMPT_TEMPLATE = """
    你是一位在制药科学领域知识渊博的助手。
    仅根据以下上下文回答问题：
    {context}

    根据上述上下文回答问题：
    {question}

    使用提供的上下文准确、简洁地回答用户的问题。
    不要为你的答案进行辩解。
    不要提供上下文信息中未提及的内容。
    不要说"根据上下文"或"上下文提到"等类似表述。
    """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # 初始化生成器（即聊天模型）
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=st.session_state.get("gemini_api_key"),
        temperature=1
    )

    # 初始化输出解析器
    output_parser = StrOutputParser()

    # RAG链
    rag_chain = {"context": retriever | format_docs,
                 "question": RunnablePassthrough()} | prompt_template | chat_model | output_parser

    # 调用链
    response = rag_chain.invoke(query)

    return response


def main():
    """初始化和管理PharmaQuery应用界面。

    此函数设置PharmaQuery（制药洞察检索系统）的Streamlit应用界面。
    用户可以输入与制药行业相关的查询，上传研究文档，并管理API密钥以增强功能。

    主要功能包括：
    - 查询输入区域，供用户提出有关制药行业的问题。
    - 提交按钮，用于处理查询并显示检索到的见解。
    - 用于API密钥输入和管理的侧边栏。
    - 文件上传器，用于将研究文档添加到数据库，增强查询响应。

    参数:
        None

    返回:
        None"""
    st.set_page_config(page_title="PharmaQuery", page_icon=":microscope:")
    st.header("制药洞察检索系统")

    query = st.text_area(
        ":bulb: 输入您关于制药行业的查询：",
        placeholder="例如：人工智能在药物发现中有哪些应用？"
    )

    if st.button("提交"):
        if not query:
            st.warning("请提出问题")

        else:
            with st.spinner("思考中..."):
                result = run_rag_chain(query=query)
                st.write(result)

    with st.sidebar:
        st.title("API密钥")
        gemini_api_key = st.text_input("输入您的Gemini API密钥：", type="password")

        if st.button("确认"):
            if gemini_api_key:
                st.session_state.gemini_api_key = gemini_api_key
                st.success("API密钥已保存！")

            else:
                st.warning("请输入您的Gemini API密钥以继续。")

    with st.sidebar:
        st.markdown("---")
        pdf_docs = st.file_uploader("上传与制药科学相关的研究文档（可选）:memo:",
                                    type=["pdf"],
                                    accept_multiple_files=True
                                    )

        if st.button("提交并处理"):
            if not pdf_docs:
                st.warning("请上传文件")

            else:
                with st.spinner("正在处理您的文档..."):
                    add_to_db(pdf_docs)
                    st.success(":file_folder: 文档已成功添加到数据库！")

    # 侧边栏页脚
    st.sidebar.write("用❤️构建 by [Charan](https://www.linkedin.com/in/codewithcharan/)")


if __name__ == "__main__":
    main()