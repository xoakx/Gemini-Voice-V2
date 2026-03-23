import os
import asyncio
import json
import base64
import websockets
from dotenv import load_dotenv

load_dotenv()

async def websocket_smoke_test():
    api_key = os.getenv("GEMINI_API_KEY")
    # v1alpha bidi endpoint
    url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService/BidiGenerateContent?key={api_key}"
    
    print(f"Connecting to {url}...", flush=True)
    async with websockets.connect(url) as ws:
        print("Connected. Sending setup...", flush=True)
        setup = {
            "setup": {
                "model": "models/gemini-2.5-flash-native-audio-latest",
                "generation_config": {
                    "response_modalities": ["AUDIO"]
                }
            }
        }
        await ws.send(json.dumps(setup))
        
        # Wait for setup response
        resp = await ws.recv()
        print(f"Setup response: {resp}", flush=True)
        
        # Send 1 second of silence
        print("Sending 1s of silence...", flush=True)
        audio_msg = {
            "realtime_input": {
                "media_chunks": [
                    {
                        "data": base64.b64encode(b'\x00' * 32000).decode('utf-8'),
                        "mime_type": "audio/pcm;rate=16000"
                    }
                ]
            }
        }
        await ws.send(json.dumps(audio_msg))
        
        # Listen for responses
        print("Waiting for any response (15s timeout)...", flush=True)
        try:
            while True:
                resp = await asyncio.wait_for(ws.recv(), timeout=15)
                print(f"Message: {resp}", flush=True)
        except asyncio.TimeoutError:
            print("Timed out.", flush=True)

if __name__ == "__main__":
    asyncio.run(websocket_smoke_test())
