from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

config = {'run_name': 'llm_test', 'tags': ['nitish', 'campusx'], "metadata": {
        "user_id": 12345,
        "source": "cli_test",
        "env": "dev"
    }}

response = model.invoke('What is the capital of Peru', config=config)

print(response.content)