import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def list_bidi_models():
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    print("Models supporting bidiGenerateContent on v1alpha:")
    for m in client.models.list():
        # Print the whole object to see exactly what we have
        print(f"Model: {m}")

if __name__ == "__main__":
    list_bidi_models()
