import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get the API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

# Initialize the LLM instance
llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-120b",
    temperature=0.7
)