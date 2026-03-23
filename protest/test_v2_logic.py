import os
import asyncio
import wave
import time
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Audio Constants
RATE = 16000
CHUNK = 2048

class AutomatedV2Tester:
    def __init__(self, audio_path):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_id = "models/gemini-2.5-flash-native-audio-latest"
        self.audio_path = audio_path
        
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})
        self.turn_complete_event = asyncio.Event()
        self.responses = []
        self.is_running = True

    async def send_audio_file(self, session, label):
        print(f"\n[Tester]: Sending Audio for {label}...")
        with wave.open(self.audio_path, 'rb') as wf:
            data = wf.readframes(CHUNK)
            count = 0
            while data:
                await session.send_realtime_input(audio=types.Blob(data=data, mime_type='audio/pcm;rate=16000'))
                data = wf.readframes(CHUNK)
                count += 1
                if count % 20 == 0: print(".", end="", flush=True)
                await asyncio.sleep(CHUNK / RATE)
        
        # Signal end of turn using the correct method
        print(" [Done]. Sending end_of_turn signal...")
        await session.send_client_content(turns=[], turn_complete=True)

    async def receiver(self, session):
        try:
            async for message in session.receive():
                print(f"\n[SERVER]: {message}")
                if message.server_content:
                    if message.server_content.model_turn:
                        for part in message.server_content.model_turn.parts:
                            if part.text:
                                print(f"[Brain]: {part.text}")
                                self.responses.append(part.text)
                    
                    if message.server_content.turn_complete:
                        print("[System]: Turn Complete Received.")
                        self.turn_complete_event.set()
                
                if message.tool_call:
                    print(f"[System]: Tool Call: {message.tool_call.function_calls[0].name}")
                    responses = []
                    for fc in message.tool_call.function_calls:
                        responses.append(types.FunctionResponse(name=fc.name, id=fc.id, response={"result": "Test Success"}))
                    await session.send_tool_response(function_responses=responses)

        except Exception as e:
            print(f"\n[Receiver Error]: {e}")
        finally:
            self.is_running = False

    async def sender_noop(self):
        # Keeps the session alive if needed, or just waits
        while self.is_running:
            await asyncio.sleep(1)

    async def run_test(self):
        config = {
            "system_instruction": "You are a concise voice assistant.",
            "response_modalities": ["AUDIO"],
            "input_audio_transcription": {},
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}}
        }

        print(f"🚀 Starting Automated Repro (Verbose, v1alpha)...")
        try:
            async with self.client.aio.live.connect(model=self.model_id, config=config) as session:
                print("✅ Session Connected. Waiting for link to stabilize...", flush=True)
                await asyncio.sleep(2) 
                
                # Start receiver
                rec_task = asyncio.create_task(self.receiver(session))
                
                # Turn 1: Try a simple text ping first
                print("\n[Tester]: Sending Text Turn 1...")
                await session.send_client_content(turns=[types.Content(parts=[types.Part(text="Hello? This is turn 1.")])], turn_complete=True)
                
                try:
                    await asyncio.wait_for(self.turn_complete_event.wait(), timeout=30)
                    print("✅ Turn 1 Verified.")
                    self.turn_complete_event.clear()
                except asyncio.TimeoutError:
                    print("❌ Turn 1 FAILED (Timeout)")
                
                # Turn 2
                await asyncio.sleep(2)
                print("\n[Tester]: Sending Turn 2...")
                await session.send_client_content(turns=[types.Content(parts=[types.Part(text="This is turn 2. Did you remember turn 1?")])], turn_complete=True)
                
                try:
                    await asyncio.wait_for(self.turn_complete_event.wait(), timeout=25)
                    print("✅ Turn 2 Verified.")
                except asyncio.TimeoutError:
                    print("❌ Turn 2 FAILED (Reproduced one-query bug)")

                self.is_running = False
                rec_task.cancel()
                
        except Exception as e:
            print(f"\n[Session Error]: {e}")

if __name__ == "__main__":
    tester = AutomatedV2Tester("/home/kms/snippet_16k_mono.wav")
    asyncio.run(tester.run_test())
