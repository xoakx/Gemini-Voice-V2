import os
import asyncio
import json
import base64
import wave
import websockets
from dotenv import load_dotenv

load_dotenv()

async def definitive_ws_test():
    api_key = os.getenv("GEMINI_API_KEY")
    # v1alpha Bidi URL
    url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={api_key}"
    
    print(f"Connecting to {url}...", flush=True)
    try:
        async with websockets.connect(url) as ws:
            print("Connected. Sending Setup...", flush=True)
            # Top-level is 'setup'
            setup = {
                "setup": {
                    "model": "models/gemini-2.5-flash-native-audio-latest",
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {"voiceName": "Aoede"}
                            }
                        }
                    }
                }
            }
            await ws.send(json.dumps(setup))
            
            resp = await ws.recv()
            print(f"Setup Response: {resp}", flush=True)
            
            # Send real audio in chunks
            print("Sending 3s of boosted audio in 0.1s chunks...", flush=True)
            with wave.open("/home/kms/test_input_boost.wav", 'rb') as wf:
                for _ in range(30): # 3 seconds
                    raw_audio = wf.readframes(1600) # 0.1s at 16k
                    audio_msg = {
                        "realtimeInput": {
                            "mediaChunks": [
                                {
                                    "data": base64.b64encode(raw_audio).decode('utf-8'),
                                    "mimeType": "audio/pcm;rate=16000"
                                }
                            ]
                        }
                    }
                    await ws.send(json.dumps(audio_msg))
                    await asyncio.sleep(0.1)
            
            # Signal End of Turn
            print("Sending Turn Complete...", flush=True)
            complete_msg = {
                "clientContent": {
                    "turnComplete": True
                }
            }
            await ws.send(json.dumps(complete_msg))
            
            # Listen
            print("Waiting for responses (20s timeout)...", flush=True)
            try:
                while True:
                    resp = await asyncio.wait_for(ws.recv(), timeout=20)
                    # Don't print full audio data, just the keys
                    msg_json = json.loads(resp)
                    print(f"\n[SERVER MESSAGE]: {list(msg_json.keys())}", flush=True)
                    if "serverContent" in msg_json:
                        content = msg_json["serverContent"]
                        if "modelTurn" in content:
                            print(f"  Model Turn Parts: {[list(p.keys()) for p in content['modelTurn']['parts']]}")
                        if "turnComplete" in content:
                            print("  Turn Complete Received.")
                            break
            except asyncio.TimeoutError:
                print("\n[System]: Response timeout.", flush=True)
            except websockets.exceptions.ConnectionClosed as e:
                print(f"\n[System]: Connection closed: {e.code} {e.reason}", flush=True)
                
    except Exception as e:
        print(f"Error: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(definitive_ws_test())
