import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

gemini_client = genai.Client(
    api_key=os.getenv("gemini_api_key")
)

config = types.GenerateContentConfig(
    system_instruction="you are an AI  mentor . please provide precise, technical answers",
    temperature=0.2,     
    top_p=0.95,
    max_output_tokens=500,
    seed=42 

     
)


response = gemini_client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Explain what RAG is",
     config=config

)

print(response.text)