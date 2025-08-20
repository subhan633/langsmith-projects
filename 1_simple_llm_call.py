from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatGroq(
    api_key=groq_api_key,
    model='openai/gpt-oss-120b',
    temperature=0.7)

parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Pakistan?"})
print(result)
