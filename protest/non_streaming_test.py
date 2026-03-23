import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def non_streaming_test():
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    model_id = "models/gemini-2.5-flash-native-audio-latest"
    print(f"Testing non-streaming with {model_id}...", flush=True)
    try:
        response = client.models.generate_content(
            model=model_id,
            contents="Say 'Hello world from non-streaming!'"
        )
        print(f"Response: {response.text}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

if __name__ == "__main__":
    non_streaming_test()
