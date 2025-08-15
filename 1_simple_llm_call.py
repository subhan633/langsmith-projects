from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

response = model.invoke('What is the capital of Peru', config=config)

print(response.content)
