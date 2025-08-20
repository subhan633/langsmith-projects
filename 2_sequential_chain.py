from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ['LANGCHAIN_PROJECT'] = 'sequential llm app'

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGroq(
    api_key=groq_api_key,
    model='openai/gpt-oss-120b',
    temperature=0.7)

model2 = ChatGroq(
    api_key=groq_api_key,
    model='openai/gpt-oss-120b',
    temperature=0.5)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config={
    'tags': ['llm ap', 'report generation', "summarization"],
    'metadata': {'models': 'openai/gpt-oss-120b', 'model_temp': '0.7, 0.5'}
}

result = chain.invoke({'topic': 'Unemployment in Pakistan'}, config=config)

print(result)
