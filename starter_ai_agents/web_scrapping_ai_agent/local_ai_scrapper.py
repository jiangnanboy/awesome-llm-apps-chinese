# Import the required libraries
import streamlit as st
from scrapegraphai.graphs import SmartScraperGraph

# Set up the Streamlit app
st.title("网页抓取人工智能代理 🕵️‍♂️")
st.caption("这个应用程序允许你使用 Llama 3.2 抓取网站")

# Set up the configuration for the SmartScraperGraph
graph_config = {
    "llm": {
        "model": "ollama/llama3.2",
        "temperature": 0,
        "format": "json",  # Ollama needs the format to be specified explicitly
        "base_url": "http://localhost:11434",  # set Ollama URL
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",
        "base_url": "http://localhost:11434",  # set Ollama URL
    },
    "verbose": True,
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
