import os
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()

async def key_test():
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"Testing Key: {api_key[:10]}...", flush=True)
    
    # Try both v1alpha and v1beta with the simplest possible model name
    for version in ['v1alpha', 'v1beta']:
        print(f"\n--- Checking version {version} ---", flush=True)
        client = genai.Client(api_key=api_key, http_options={'api_version': version})
        try:
            # Try to list some models to see what we CAN see
            models = list(client.models.list())
            print(f"Can list {len(models)} models.", flush=True)
            if models:
                first_model = models[0].name
                print(f"First available model: {first_model}", flush=True)
                # Try a standard request with the first available model
                try:
                    resp = client.models.generate_content(model=first_model, contents="Ping")
                    print(f"SUCCESS: {first_model} responded: {resp.text.strip()}", flush=True)
                except Exception as e:
                    print(f"FAIL: {first_model} failed generate_content: {e}", flush=True)
        except Exception as e:
            print(f"ERROR: Could not list models on {version}: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(key_test())
