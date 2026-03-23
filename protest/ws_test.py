import os
import asyncio
import json
import base64
import websockets
from dotenv import load_dotenv

load_dotenv()

async def ws_test():
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService/BidiGenerateContent?key={api_key}"
    
    print(f"Connecting to {url}...", flush=True)
    try:
        async with websockets.connect(url) as ws:
            print("Connected. Sending Setup...", flush=True)
            setup = {
                "setup": {
                    "model": "models/gemini-2.5-flash-native-audio-latest",
                    "generation_config": {
                        "response_modalities": ["AUDIO"]
                    }
                }
            }
            await ws.send(json.dumps(setup))
            
            # Response 1: Setup Response
            resp = await ws.recv()
            print(f"Setup Response: {resp}", flush=True)
            
            # Send audio
            print("Sending 3s of silence...", flush=True)
            audio = {
                "realtime_input": {
                    "media_chunks": [
                        {
                            "data": base64.b64encode(b'\x00' * 32000).decode('utf-8'),
                            "mime_type": "audio/pcm;rate=16000"
                        }
                    ]
                }
            }
            await ws.send(json.dumps(audio))
            
            # Listen
            while True:
                resp = await asyncio.wait_for(ws.recv(), timeout=10)
                print(f"Message: {resp}", flush=True)
                
    except Exception as e:
        print(f"Error: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(ws_test())
