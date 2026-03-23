import os
import sys
import json
import asyncio
import sounddevice as sd
import numpy as np
import threading
import queue
import traceback
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Audio Constants
IN_RATE = 16000
OUT_RATE = 24000
CHANNELS = 1
CHUNK = 2048

class CoreTestAssistant:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_version = 'v1alpha'
        self.model_id = "gemini-2.5-flash-native-audio-latest"

        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': self.api_version})

        self.mic_queue = queue.Queue(maxsize=50)
        self.speaker_queue = queue.Queue()
        self.is_running = True
        
        # Load existing config
        config_path = os.path.join(os.path.dirname(__file__), "../../config/audio_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            # Use PulseAudio names to find IDs for this run
            self.input_id = self._find_id(config["input_pulse_name"])
            self.output_id = self._find_id(config["output_pulse_name"])

    def _find_id(self, name):
        for i, d in enumerate(sd.query_devices()):
            if name.lower() in d['name'].lower(): return i
        return None

    def audio_callback(self, indata, frames, time_info, status):
        if not self.is_running: return
        audio_int16 = (indata * 32767).astype(np.int16).tobytes()
        try:
            self.mic_queue.put_nowait(audio_int16)
        except queue.Full:
            try:
                self.mic_queue.get_nowait()
                self.mic_queue.put_nowait(audio_int16)
            except: pass

    def speaker_worker(self):
        with sd.OutputStream(samplerate=OUT_RATE, channels=CHANNELS, dtype='int16', device=self.output_id) as stream:
            while self.is_running:
                try:
                    data = self.speaker_queue.get(timeout=0.05)
                    stream.write(data)
                except queue.Empty: continue
                except Exception: pass

    async def main_loop(self, session):
        print("System: Link established. Multimodal core active.")

        async def receiver():
            try:
                async for response in session.receive():
                    if not self.is_running: break

                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data:
                                self.speaker_queue.put(np.frombuffer(part.inline_data.data, dtype='int16'))
                            if part.text:
                                print(f"\n[Brain]: {part.text}")

                    if response.server_content and response.server_content.interrupted:
                        print("\n[AI Interrupted] Flushing speaker buffer...")
                        while not self.speaker_queue.empty():
                            try: self.speaker_queue.get_nowait()
                            except: break

                    if response.server_content and response.server_content.turn_complete:
                        print("\n--- Listening ---")
                        # Flush mic queue to clear any echo from the previous response
                        while not self.mic_queue.empty():
                            try: self.mic_queue.get_nowait()
                            except: break

            except Exception as e:
                print(f"\n[Receiver Error]: {e}")

        async def sender():
            try:
                while self.is_running:
                    try:
                        audio_bytes = await asyncio.to_thread(self.mic_queue.get, timeout=0.1)
                        
                        # Half-Duplex Fix: Don't send mic audio if the speaker is playing
                        if not self.speaker_queue.empty():
                            continue 
                            
                        await session.send_realtime_input(
                            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                        )
                        await asyncio.sleep(0)
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                    except Exception as e:
                        print(f"\n[Sender Error]: {e}")
                        break
            except Exception as e:
                print(f"\n[Sender Task Error]: {e}")

        # Run both tasks, and finish when the first one dies
        done, pending = await asyncio.wait(
            [asyncio.create_task(sender()), asyncio.create_task(receiver())],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        print("System: Core teardown.")

    async def run(self):
        threading.Thread(target=self.speaker_worker, daemon=True).start()

        config = {
            "system_instruction": "You are a concise voice assistant. Respond directly via audio.",
            "response_modalities": ["AUDIO"],
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}}
        }

        while self.is_running:
            print(f"\n🚀 Connecting to Gemini Live Core...")
            try:
                async with self.client.aio.live.connect(model=self.model_id, config=config) as session:
                    with sd.InputStream(samplerate=IN_RATE, channels=CHANNELS, callback=self.audio_callback,
                                      blocksize=CHUNK, dtype='float32', device=self.input_id):
                        await self.main_loop(session)
            except Exception as e:
                print(f"\n[Connection Dropped]: {e}")
                if not self.is_running: break
                await asyncio.sleep(2)

if __name__ == "__main__":
    assistant = CoreTestAssistant()
    try: asyncio.run(assistant.run())
    except KeyboardInterrupt: pass
    finally: assistant.is_running = False
