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
        print(f"System: Starting Headless Test with {self.audio_path}")
        
        async def receiver():
            async for message in session.receive():
                if message.server_content and message.server_content.model_turn:
                    for part in message.server_content.model_turn.parts:
                        if part.text:
                            print(f"\n[Brain]: {part.text}")
                        if part.inline_data:
                            # We just acknowledge receipt of audio
                            pass
                    self.responses_received += 1
                    print(f"\n[System]: Response {self.responses_received} received.")

                if message.server_content and message.server_content.turn_complete:
                    print("\n[System]: Turn Complete.")
                    self.turn_complete_event.set()

        async def sender():
            # Turn 1
            print("\n[System]: Sending Turn 1...")
            await self.send_audio(session)
            
            # Wait for turn to complete
            try:
                await asyncio.wait_for(self.turn_complete_event.wait(), timeout=15)
                self.turn_complete_event.clear()
            except asyncio.TimeoutError:
                print("\n[Error]: Timeout waiting for Turn 1 response.")
                return

            print("\n[System]: Waiting 3 seconds before Turn 2...")
            await asyncio.sleep(3)

            # Turn 2
            print("\n[System]: Sending Turn 2...")
            await self.send_audio(session)
            
            # Wait for turn to complete
            try:
                await asyncio.wait_for(self.turn_complete_event.wait(), timeout=15)
            except asyncio.TimeoutError:
                print("\n[Error]: Timeout waiting for Turn 2 response.")
            
            print(f"\n[Summary]: Total responses received: {self.responses_received}")

        await asyncio.gather(sender(), receiver())

    async def send_audio(self, session):
        with wave.open(self.audio_path, 'rb') as wf:
            data = wf.readframes(CHUNK)
            while data:
                await session.send_realtime_input(audio=types.Blob(data=data, mime_type='audio/pcm;rate=16000'))
                data = wf.readframes(CHUNK)
                await asyncio.sleep(CHUNK / RATE) # Simulate real-time speed

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
