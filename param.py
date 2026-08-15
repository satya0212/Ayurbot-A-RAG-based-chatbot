import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

endpoint = HuggingFaceEndpoint(
    repo_id="Prady029/AyurParam-2.9b-it-gguf",
    task="conversational",
    temperature=0.5,
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=100,
    provider="auto",
    timeout=30,  # fail fast instead of hanging forever
)
llm = ChatHuggingFace(llm=endpoint)

print("Sending test request...")
try:
    response = llm.invoke("Hello, who are you?")
    print("SUCCESS:", response.content)
except Exception as e:
    print("FAILED:", type(e).__name__, "-", e)