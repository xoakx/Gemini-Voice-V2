import os
import sys
import json
import asyncio
import sounddevice as sd
import numpy as np
import threading
import queue
import ollama
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Audio Constants - Increased for network stability
IN_RATE = 16000
OUT_RATE = 24000
CHANNELS = 1
CHUNK = 4096 

class LiveToolAssistant:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_version = 'v1alpha'
        self.model_id = "gemini-2.5-flash-native-audio-latest"
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': self.api_version})

        self.mic_queue = queue.Queue(maxsize=20) # Small queue to prevent lag
        self.speaker_queue = queue.Queue()
        self.is_running = True
        self.assistant_speaking = False
        self._load_config()

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "../../config/audio_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            devices = sd.query_devices()
            self.input_id = next((i for i, d in enumerate(devices) if config["input_pulse_name"].lower() in d['name'].lower() and d['max_input_channels'] > 0), None)
            self.output_id = next((i for i, d in enumerate(devices) if config["output_pulse_name"].lower() in d['name'].lower() and d['max_output_channels'] > 0), None)

    def audio_callback(self, indata, frames, time_info, status):
        """Mic -> Queue (Skip if Assistant is talking)"""
        if not self.is_running or self.assistant_speaking: return
        audio_int16 = (indata * 32767).astype(np.int16).tobytes()
        try:
            self.mic_queue.put_nowait(audio_int16)
        except queue.Full:
            pass # Drop frames if network is slow to prevent 'glitchy' backlog

    def speaker_worker(self):
        """Queue -> Speaker (Continuous output)"""
        # Lower latency output stream
        with sd.OutputStream(samplerate=OUT_RATE, channels=CHANNELS, dtype='int16', device=self.output_id, blocksize=CHUNK//2) as stream:
            while self.is_running:
                try:
                    data = self.speaker_queue.get(timeout=0.1)
                    stream.write(data)
                except queue.Empty: continue
                except Exception: pass

    async def main_loop(self, session):
        print("System: Conversation link established.")

        async def receiver():
            try:
                async for response in session.receive():
                    if not self.is_running: break

                    if response.server_content and response.server_content.model_turn:
                        self.assistant_speaking = True
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data:
                                # Queue audio chunks instantly
                                audio_chunk = np.frombuffer(part.inline_data.data, dtype='int16')
                                self.speaker_queue.put(audio_chunk)
                            if part.text:
                                print(f"\n[Brain]: {part.text}")

                    if response.server_content and (response.server_content.interrupted or response.server_content.turn_complete):
                        if response.server_content.interrupted:
                            # Kill current playback immediately
                            while not self.speaker_queue.empty():
                                try: self.speaker_queue.get_nowait()
                                except: break
                        
                        # Wait for audio to actually finish playing before re-enabling mic
                        await asyncio.sleep(0.5) 
                        self.assistant_speaking = False
                        
                        if response.server_content.turn_complete:
                            print("\n--- Listening ---")

                    if response.tool_call:
                        # (Ollama tool logic remains the same)
                        function_responses = []
                        for fc in response.tool_call.function_calls:
                            if fc.name == "query_local_brain":
                                print("\n[Gemini]: Querying Local Brain...")
                                try:
                                    result = await asyncio.to_thread(ollama.chat, model='gemma2:9b', messages=[{'role': 'user', 'content': fc.args.get("prompt")}])
                                    function_responses.append(types.FunctionResponse(name=fc.name, id=fc.id, response={"result": result['message']['content']}))
                                except Exception as e:
                                    function_responses.append(types.FunctionResponse(name=fc.name, id=fc.id, response={"result": f"Local error: {e}"}))
                        await session.send_tool_response(function_responses=function_responses)

            except Exception as e:
                print(f"\n[Receiver Error]: {e}")

        async def sender():
            try:
                while self.is_running:
                    try:
                        audio_bytes = await asyncio.to_thread(self.mic_queue.get, timeout=0.1)
                        await session.send_realtime_input(audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000"))
                        await asyncio.sleep(0) # Heartbeat yield
                    except queue.Empty: await asyncio.sleep(0.01)
            except Exception as e:
                print(f"\n[Sender Error]: {e}")

        # Run both tasks, and finish when the first one dies
        done, pending = await asyncio.wait(
            [asyncio.create_task(sender()), asyncio.create_task(receiver())],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        print("System: Conversation teardown.")

    async def run(self):
        threading.Thread(target=self.speaker_worker, daemon=True).start()
        
        local_brain_tool = types.Tool(function_declarations=[types.FunctionDeclaration(
            name="query_local_brain",
            description="Use local Gemma 2 9b for system admin and hardware status.",
            parameters={"type": "OBJECT", "properties": {"prompt": {"type": "STRING"}}, "required": ["prompt"]}
        )])

        config = {
            "system_instruction": "You are a concise voice assistant. Respond via audio.",
            "response_modalities": ["AUDIO"],
            "tools": [local_brain_tool],
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}}
        }

        while self.is_running:
            print(f"\n🚀 Launching Smooth Link ({self.model_id})...")
            try:
                async with self.client.aio.live.connect(model=self.model_id, config=config) as session:
                    print("✅ LINK STABLE.")
                    with sd.InputStream(samplerate=IN_RATE, channels=CHANNELS, callback=self.audio_callback,
                                      blocksize=CHUNK, dtype='float32', device=self.input_id):
                        await self.main_loop(session)
            except Exception as e:
                print(f"\n[Drop]: {e}")
                await asyncio.sleep(2)

if __name__ == "__main__":
    assistant = LiveToolAssistant()
    try: asyncio.run(assistant.run())
    except KeyboardInterrupt: pass
    finally: assistant.is_running = False
