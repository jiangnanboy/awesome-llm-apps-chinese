from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from google.generativeai import upload_file, get_file
import time

# 1. Initialize the Multimodal Agent
agent = Agent(model=Gemini(id="gemini-2.0-flash-exp"), tools=[DuckDuckGoTools()], markdown=True)

# 2. Image Input
image_url = "https://example.com/sample_image.jpg"

# 3. Audio Input
audio_file = "sample_audio.mp3"

# 4. Video Input
video_file = upload_file("sample_video.mp4")  
while video_file.state.name == "PROCESSING":  
    time.sleep(2)
    video_file = get_file(video_file.name)

# 5. Multimodal Query
query = """ 
结合来自以下输入的洞察：
图像：描述场景及其意义。
音频：提取与视觉相关的关键信息。
视频：查看视频输入，并提供与图像和音频背景相关的洞察。
网络搜索：查找与所有这些主题相关的最新更新或事件。
总结这些输入所传达的整体主题或故事。
"""

# 6. Multimodal Agent generates unified response
agent.print_response(query, images=[image_url], audio=audio_file, videos=[video_file], stream=True)