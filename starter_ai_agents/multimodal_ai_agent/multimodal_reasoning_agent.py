import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
import tempfile
import os

def main():
    # Set up the reasoning agent
    agent = Agent(
        model=Gemini(id="gemini-2.0-flash-thinking-exp-1219"), 
        markdown=True
    )

    # Streamlit app title
    st.title("多模态推理人工智能代理 🧠")

    # Instruction
    st.write(
        "上传一张图片，并为人工智能代理提供一个基于推理的任务。人工智能代理将分析图片，并根据你的输入做出回应。"
    )

    # File uploader for image
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Save uploaded file to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name

            # Display the uploaded image
            st.image(uploaded_file, caption="上传图片", use_container_width=True)

            # Input for dynamic task
            task_input = st.text_area(
                "输入你对人工智能代理的任务 / 问题："
            )

            # Button to process the image and task
            if st.button("分析图像") and task_input:
                with st.spinner("AI正在思考... 🤖"):
                    try:
                        # Call the agent with the dynamic task and image path
                        response = agent.run(task_input, images=[temp_path])
                        
                        # Display the response from the model
                        st.markdown("### AI 回复:")
                        st.markdown(response.content)
                    except Exception as e:
                        st.error(f"分析过程中发生了错误: {str(e)}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

        except Exception as e:
            st.error(f"处理图像时发生了错误: {str(e)}")

if __name__ == "__main__":
    main()