import os
import json
import asyncio
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import traceback
import ollama
import concurrent.futures
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Audio Constants
RATE = 16000
CHANNELS = 1
CHUNK = 2048

class ProductionV2Assistant:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_version = 'v1alpha'
        self.model_id = "gemini-2.5-flash-native-audio-latest"
        
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': self.api_version})
        
        self.mic_queue = asyncio.Queue()
        self.speaker_queue = queue.Queue()
        self.is_running = True
        self.assistant_speaking = False
        
        # Load audio config
        config_path = os.path.join(os.path.dirname(__file__), "../../config/audio_config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)
            self.input_id = self._find_id(cfg["input_pulse_name"])
            self.output_id = self._find_id(cfg["output_pulse_name"])

    def _find_id(self, name):
        for i, d in enumerate(sd.query_devices()):
            if name.lower() in d['name'].lower(): return i
        return None

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Duplex Hardware Bridge."""
        if not self.is_running: return
        
        # Mic In
        audio_bytes = indata.copy().tobytes()
        if hasattr(self, 'loop'):
            self.loop.call_soon_threadsafe(self.mic_queue.put_nowait, audio_bytes)
        
        # Speaker Out
        try:
            data = self.speaker_queue.get_nowait()
            outdata[:len(data)] = data.reshape(-1, 1)
            if len(data) < len(outdata): outdata[len(data):] = 0
        except queue.Empty:
            outdata.fill(0)

    async def main_loop(self, session):
        print("System: Unified Link Active.")
        
        async def receiver():
            try:
                async for message in session.receive():
                    if not self.is_running: break
                    
                    # 1. Handle Transcripts
                    if message.server_content and message.server_content.input_transcription:
                        print(f"\rUser: {message.server_content.input_transcription.text}", end='      \n')

                    # 2. Handle Audio
                    if message.server_content and message.server_content.model_turn:
                        for part in message.server_content.model_turn.parts:
                            if part.inline_data:
                                self.speaker_queue.put(np.frombuffer(part.inline_data.data, dtype='int16'))
                            if part.text:
                                print(f"\n[Brain]: {part.text}")

                    # 3. Handle Tool Calls (Local Logic)
                    if message.tool_call:
                        responses = []
                        for fc in message.tool_call.function_calls:
                            if fc.name == "query_local_brain":
                                print("\n[System]: Consulting Local Brain (Gemma 2 9b)...")
                                try:
                                    result = await asyncio.to_thread(ollama.chat, model='gemma2:9b', 
                                                                   messages=[{'role': 'user', 'content': fc.args.get("prompt")}])
                                    responses.append(types.FunctionResponse(name=fc.name, id=fc.id, response={"result": result['message']['content']}))
                                except Exception as e:
                                    responses.append(types.FunctionResponse(name=fc.name, id=fc.id, response={"result": f"Local Logic Error: {e}"}))
                        await session.send_tool_response(function_responses=responses)
            except Exception as e:
                print(f"\n[Receiver Drop]: {e}")

        async def sender():
            try:
                while self.is_running:
                    audio_bytes = await self.mic_queue.get()
                    await session.send_realtime_input(audio=types.Blob(data=audio_bytes, mime_type='audio/pcm;rate=16000'))
            except Exception as e:
                print(f"\n[Sender Drop]: {e}")

        # Run both tasks, and finish when the first one dies
        done, pending = await asyncio.wait(
            [asyncio.create_task(sender()), asyncio.create_task(receiver())],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        print("System: Session teardown complete.")

    async def run(self):
        self.loop = asyncio.get_running_loop()
        # Define the plugin
        local_tool = types.Tool(function_declarations=[types.FunctionDeclaration(
            name="query_local_brain",
            description="Query local system status, bash commands, or hardware specs via Gemma 2 9b.",
            parameters={"type": "OBJECT", "properties": {"prompt": {"type": "STRING"}}, "required": ["prompt"]}
        )])

        config = {
            "system_instruction": "You are a concise voice assistant. Use the query_local_brain tool for system-specific questions. Respond only via audio.",
            "response_modalities": ["AUDIO"],
            "tools": [local_tool],
            "input_audio_transcription": {},
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}}
        }
        
        while self.is_running:
            print(f"\n🚀 Establishing Secure Link...")
            try:
                async with self.client.aio.live.connect(model=self.model_id, config=config) as session:
                    with sd.Stream(device=(self.input_id, self.output_id), samplerate=RATE, 
                                  channels=CHANNELS, dtype='int16', callback=self.audio_callback, blocksize=CHUNK):
                        await self.main_loop(session)
            except Exception as e:
                print(f"\n[Link Drop]: {e}")
                await asyncio.sleep(3)

if __name__ == "__main__":
    assistant = ProductionV2Assistant()
    try: asyncio.run(assistant.run())
    except KeyboardInterrupt: pass
    finally: assistant.is_running = False
