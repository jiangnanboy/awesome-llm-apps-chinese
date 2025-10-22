# Import the required libraries
import streamlit as st
from scrapegraphai.graphs import SmartScraperGraph

# Set up the Streamlit app
st.title("网页抓取人工智能代理 🕵️‍♂️")
st.caption("这款应用程序允许你使用 OpenAI API 抓取网站")

# Get OpenAI API key from user
openai_access_token = st.text_input("OpenAI API Key", type="password")

if openai_access_token:
    model = st.radio(
        "选择模型",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0,
    )    
    graph_config = {
        "llm": {
            "api_key": openai_access_token,
            "model": model,
        },
    }
    # Get the URL of the website to scrape
    url = st.text_input("输入你想要抓取的网站的 URL")
    # Get the user prompt
    user_prompt = st.text_input("你希望人工智能代理从网站上抓取什么内容？")
    
    # Create a SmartScraperGraph object
    smart_scraper_graph = SmartScraperGraph(
        prompt=user_prompt,
        source=url,
        config=graph_config
    )
    # Scrape the website
    if st.button("抓取"):
        result = smart_scraper_graph.run()
        st.write(result)