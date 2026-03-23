import os
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()

async def test_connection(model_id, api_version):
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': api_version})
    config = {"response_modalities": ["AUDIO"]}
    
    print(f"Testing: Model='{model_id}', API='{api_version}'...")
    try:
        # We use a timeout so it doesn't hang if it partially connects
        async with asyncio.timeout(5):
            async with client.aio.live.connect(model=model_id, config=config) as session:
                print(f"✅ SUCCESS: {model_id} works with {api_version}!")
                return True
    except asyncio.TimeoutError:
        print(f"⚠️ TIMEOUT: {model_id} with {api_version} (Might work, but slow)")
        return False
    except Exception as e:
        # print(f"❌ FAILED: {e}")
        return False

async def main():
    # Combinations to test based on your ListModels output
    tests = [
        ("gemini-2.0-flash", "v1alpha"),
        ("gemini-2.0-flash", "v1beta"),
        ("gemini-2.5-flash-native-audio-latest", "v1alpha"),
        ("gemini-2.0-flash-exp", "v1alpha"),
        ("gemini-2.0-flash", "v1"),
    ]
    
    for model, api in tests:
        if await test_connection(model, api):
            print(f"\n🚀 FINAL VERDICT: Use Model='{model}' and API='{api}'")
            return

    print("\n❌ All standard combinations failed. Checking if 'models/' prefix is required...")
    if await test_connection("models/gemini-2.0-flash", "v1alpha"):
        print("\n🚀 FINAL VERDICT: Use Model='models/gemini-2.0-flash' and API='v1alpha'")

if __name__ == "__main__":
    asyncio.run(main())
