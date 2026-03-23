import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def list_models():
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    print("Models on v1alpha:")
    for m in client.models.list():
        if "flash" in m.name.lower():
            print(f" - {m.name}")

if __name__ == "__main__":
    list_models()
