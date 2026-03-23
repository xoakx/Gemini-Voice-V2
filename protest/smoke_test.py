import os
import asyncio
import wave
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

async def smoke_test():
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"API Key: {api_key[:8]}...", flush=True)
    
    # Use exact same setup as production live_audio.py
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    model_id = "models/gemini-2.5-flash"
    
    config = {
        "system_instruction": "You are a concise voice assistant. Respond only with 'Turn 1 success' if you hear me.",
        "response_modalities": ["AUDIO"],
        "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}}
    }
    
    print(f"Connecting to {model_id} via v1alpha...", flush=True)
    try:
        async with client.aio.live.connect(model=model_id, config=config) as session:
            print("Connected. Link Established.", flush=True)
            
            async def receiver():
                print("Receiver monitoring...", flush=True)
                async for message in session.receive():
                    print(f"\n[SERVER]: {message}", flush=True)
                    if message.server_content and message.server_content.turn_complete:
                        print("Turn complete received!", flush=True)

            async def sender():
                print("Sender starting. Sending boosted audio snippet...", flush=True)
                with wave.open("/home/kms/test_input_boost.wav", 'rb') as wf:
                    # Send 3s of audio
                    data = wf.readframes(16000 * 3)
                    chunk_size = 1600
                    for i in range(0, len(data), chunk_size * 2):
                        chunk = data[i:i+chunk_size*2]
                        await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type='audio/pcm;rate=16000'))
                        await asyncio.sleep(0.1)
                print("Audio sent. Waiting 15s for processing...", flush=True)
                await asyncio.sleep(15)

            await asyncio.gather(receiver(), sender())
    except Exception as e:
        print(f"Error: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(smoke_test())
