import os
import asyncio
import wave
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

async def manual_test_24k():
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    model_id = "models/gemini-2.5-flash-native-audio-latest"
    
    config = {
        "system_instruction": "Say 'OK' if you hear me.",
        "response_modalities": ["AUDIO"],
        "input_audio_transcription": {}
    }
    
    print(f"Connecting to {model_id}...", flush=True)
    try:
        async with client.aio.live.connect(model=model_id, config=config) as session:
            print("Connected. Sending 3s of 24kHz audio...", flush=True)
            with wave.open("/home/kms/test_input_24k.wav", 'rb') as wf:
                data = wf.readframes(24000 * 3)
                await session.send_realtime_input(audio=types.Blob(data=data, mime_type='audio/pcm;rate=24000'))
            print("Sent. Waiting for server...", flush=True)
            
            async def receiver():
                async for message in session.receive():
                    print(f"Message: {message}", flush=True)
                    if message.server_content and message.server_content.turn_complete:
                        print("Turn complete received!", flush=True)
                        return

            try:
                await asyncio.wait_for(receiver(), timeout=15)
            except asyncio.TimeoutError:
                print("Timeout waiting for server response.", flush=True)
                
    except Exception as e:
        print(f"Error: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(manual_test_24k())
