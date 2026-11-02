import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

models = client.models.list()

print("API connection successful!")
print("\nAvailable models:")

for model in sorted(models.data, key=lambda m: m.id):
    print(model.id)
