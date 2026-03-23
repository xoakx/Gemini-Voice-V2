import os
import asyncio
import wave
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Audio Constants
RATE = 16000
CHUNK = 2048

class HeadlessTester:
    def __init__(self, audio_path):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_version = 'v1alpha'
        self.model_id = "gemini-2.5-flash-native-audio-latest"
        self.audio_path = audio_path
        
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': self.api_version})
        self.responses_received = 0
        self.turn_complete_event = asyncio.Event()

    async def main_loop(self, session):
        print(f"System: Starting Headless Test with {self.audio_path}", flush=True)
        
        receiver_task = asyncio.create_task(self.receiver(session))
        sender_task = asyncio.create_task(self.sender(session))
        
        # Run until sender is done (it handles both turns)
        done, pending = await asyncio.wait(
            [sender_task], 
            timeout=60 # Global test timeout
        )
        
        # Cleanup
        for task in [sender_task, receiver_task]:
            if not task.done():
                task.cancel()
        
        print(f"\n[Summary]: Total responses received: {self.responses_received}", flush=True)

    async def receiver(self, session):
        try:
            async for message in session.receive():
                print(f"\n[Raw Message]: {type(message)}", flush=True)
                if message.server_content:
                    print(f"[Content]: {message.server_content}", flush=True)
                    if message.server_content.model_turn:
                        for part in message.server_content.model_turn.parts:
                            if part.text:
                                print(f"\n[Brain]: {part.text}", flush=True)
                        self.responses_received += 1
                        print(f"\n[System]: Response {self.responses_received} received.", flush=True)

                    if message.server_content.turn_complete:
                        print("\n[System]: Turn Complete.", flush=True)
                        self.turn_complete_event.set()
                
                if message.tool_call:
                    print(f"[Tool Call]: {message.tool_call}", flush=True)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\n[Receiver Error]: {e}", flush=True)

    async def sender(self, session):
        try:
            # Turn 1
            print("\n[System]: Sending Turn 1...", flush=True)
            await self.send_audio(session)
            
            # Wait for turn to complete
            print("[System]: Waiting for Turn 1 response...", flush=True)
            try:
                await asyncio.wait_for(self.turn_complete_event.wait(), timeout=20)
                self.turn_complete_event.clear()
            except asyncio.TimeoutError:
                print("\n[Error]: Timeout waiting for Turn 1 response.", flush=True)
                return

            print("\n[System]: Waiting 2 seconds before Turn 2...", flush=True)
            await asyncio.sleep(2)

            # Turn 2
            print("\n[System]: Sending Turn 2...", flush=True)
            await self.send_audio(session)
            
            # Wait for turn to complete
            print("[System]: Waiting for Turn 2 response...", flush=True)
            try:
                await asyncio.wait_for(self.turn_complete_event.wait(), timeout=20)
            except asyncio.TimeoutError:
                print("\n[Error]: Timeout waiting for Turn 2 response (This is the bug).", flush=True)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\n[Sender Error]: {e}", flush=True)

    async def send_audio(self, session):
        with wave.open(self.audio_path, 'rb') as wf:
            data = wf.readframes(CHUNK)
            count = 0
            while data:
                await session.send_realtime_input(audio=types.Blob(data=data, mime_type='audio/pcm;rate=16000'))
                data = wf.readframes(CHUNK)
                count += 1
                if count % 10 == 0:
                    print(".", end="", flush=True)
                await asyncio.sleep(CHUNK / (RATE * 4)) # Send 4x faster than real-time for testing
            print(" [Done]", flush=True)

    async def run(self):
        config = {
            "system_instruction": "You are a concise voice assistant. Respond only via text (since I am in headless mode).",
            "response_modalities": ["AUDIO"], # Still ask for audio to match production
            "input_audio_transcription": {},
        }
        
        try:
            async with self.client.aio.live.connect(model=self.model_id, config=config) as session:
                await self.main_loop(session)
        except Exception as e:
            print(f"\n[Link Drop]: {e}")

if __name__ == "__main__":
    tester = HeadlessTester("/home/kms/test_input.wav")
    asyncio.run(tester.run())
