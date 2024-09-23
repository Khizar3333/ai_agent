import os
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_openai import ChatOpenAI
# from langsmith import pull_prompt  # Updated SDK
# from langchain.prompts import load_prompt
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from streamlit_chat import message
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from typing import Any  # Add this import
import asyncio
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# Load environment variables
load_dotenv()
key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = ChatOpenAI(api_key=key, model="gpt-3.5-turbo", temperature=0)

# Initialize tools
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
db = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = db.as_retriever()
retriever_tool = create_retriever_tool(retriever, "langsmith_search", "Search for information on LangSmith.")

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

tools = [wiki, arxiv, retriever_tool]

# Load the agent prompt using the updated method
# prompt = load_prompt("hwchase17/openai-functions-agent")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
# Create the agent and agent executor
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Streamlit app
st.set_page_config(page_title="AI Agent Assistant", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("AI Agent Assistant")
    st.markdown("Ask questions and get answers from the AI agent.")
    
    # Add any additional sidebar elements here, e.g.:
    st.markdown("### About")
    st.markdown("This AI assistant uses Wikipedia, arXiv, and LangSmith data to answer your questions.")

# Main chat interface
st.header("Chat with AI Agent")

# Display chat messages
for i, (role, content) in enumerate(st.session_state.messages):
    message(content, is_user=role == "user", key=f"{i}_{role}")

# User input
user_input = st.chat_input("Type your question here:")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append(("user", user_input))
    message(user_input, is_user=True)
    
    # Create a placeholder for the AI response
    response_placeholder = st.empty()

    # Create a StreamlitCallbackHandler instance
    streamlit_handler = StreamlitCallbackHandler(response_placeholder)

    # Run the agent with the Streamlit callback
    with st.spinner("AI is thinking..."):
        response = agent_executor.invoke(
            {"input": user_input},
            callbacks=[streamlit_handler]
        )
    
    # Display the final response
    message(response['output'], is_user=False)
    
    # Add AI response to chat history
    st.session_state.messages.append(("assistant", response['output']))

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

