import os
import sys
import json
import asyncio
import time
import sounddevice as sd
import numpy as np
import threading
import queue
import ollama
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Audio Constants
IN_RATE = 16000
OUT_RATE = 24000
CHANNELS = 1
CHUNK = 2048
CONFIG_FILE = "audio_config.json"
MEMORY_FILE = "memory.json"

class HybridLiveAssistant:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY missing from environment.")

        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})
        self.model_id = "gemini-2.5-flash-native-audio-preview-12-2025"

        self.mic_queue = queue.Queue(maxsize=50)
        self.speaker_queue = queue.Queue()
        self.is_running = True
        self.is_playing = False # Track hardware playback state

        self._load_or_create_config()
        self._initialize_memory()

    def _initialize_memory(self):
        if not os.path.exists(MEMORY_FILE):
            default_state = {
                "assistant_profile": {
                    "purpose": "Provide low-latency voice assistance and execute local Kubuntu administration tasks."
                },
                "system_environment": {
                    "os_version": "Kubuntu 25.10",
                    "hardware_specs": {
                        "cpu": "Intel Ultra 7 265k",
                        "ram": "64GB DDR5 6000 CL28",
                        "motherboard": "MSI Z890-S"
                    }
                },
                "current_focus": {"active_projects": []}
            }
            with open(MEMORY_FILE, "w") as f:
                json.dump(default_state, f, indent=2)

    def _load_or_create_config(self):
        if not os.path.exists(CONFIG_FILE):
            print("\n--- Audio Device Setup ---")
            print(sd.query_devices())
            try:
                in_idx = int(input("Enter the ID of your preferred INPUT (Mic) device: "))
                out_idx = int(input("Enter the ID of your preferred OUTPUT (Speaker) device: "))
            except ValueError:
                print("Invalid input. Please run again.")
                sys.exit(1)

            with open(CONFIG_FILE, "w") as f:
                json.dump({"input_id": in_idx, "output_id": out_idx}, f)
            print("Configuration saved.")

        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            self.input_id = config.get("input_id")
            self.output_id = config.get("output_id")

    def audio_callback(self, indata, frames, time_info, status):
        if not self.is_running: return
        audio_int16 = (indata * 32767).astype(np.int16).tobytes()
        try:
            self.mic_queue.put_nowait(audio_int16)
        except queue.Full:
            try:
                self.mic_queue.get_nowait()
                self.mic_queue.put_nowait(audio_int16)
            except queue.Empty: pass

    def speaker_worker(self):
        """Dynamically opens/closes the audio stream to prevent PipeWire suspension."""
        while self.is_running:
            try:
                # Block until AI sends audio
                data = self.speaker_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.is_playing = True
            try:
                with sd.OutputStream(samplerate=OUT_RATE, channels=CHANNELS, dtype='int16', device=self.output_id) as stream:
                    stream.write(data)
                    # Keep reading from queue while stream is active
                    while self.is_running:
                        try:
                            next_data = self.speaker_queue.get(timeout=0.2)
                            stream.write(next_data)
                        except queue.Empty:
                            break # Stream starves safely and closes
            except Exception as e:
                print(f"\n[Speaker Error]: {e}")
            finally:
                # Wait for room echo to bounce and dissipate
                time.sleep(0.3)
                self.is_playing = False

                # Flush mic data built up during playback
                while not self.mic_queue.empty():
                    try: self.mic_queue.get_nowait()
                    except queue.Empty: break

    async def execute_tool(self, fc):
        """Handles local logic and memory file interactions."""
        if fc.name == "query_local_brain":
            prompt_arg = fc.args.get("prompt")
            try:
                result = await asyncio.to_thread(
                    ollama.chat,
                    model='gemma2:9b',
                    messages=[{'role': 'user', 'content': prompt_arg}]
                )
                return result['message']['content']
            except Exception as e:
                return f"Local logic unavailable. Error: {e}"

        elif fc.name == "read_memory":
            with open(MEMORY_FILE, "r") as f:
                return f.read()

        elif fc.name == "update_memory":
            category = fc.args.get("category")
            key = fc.args.get("key")
            value = fc.args.get("value")

            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)

            if category not in data: data[category] = {}
            if isinstance(data.get(category, {}).get(key), list):
                data[category][key].append(value)
            else:
                data[category][key] = value

            with open(MEMORY_FILE, "w") as f:
                json.dump(data, f, indent=2)
            return f"Updated {category} -> {key} with {value}."

    async def main_loop(self, session):
        print("System: Assistant is live and listening.")

        async def receiver():
            try:
                async for response in session.receive():
                    if not self.is_running: break

                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data:
                                self.speaker_queue.put(np.frombuffer(part.inline_data.data, dtype='int16'))

                    if response.tool_call:
                        function_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"\n[Gemini]: Executing local tool -> {fc.name}...")
                            reply = await self.execute_tool(fc)
                            function_responses.append(types.FunctionResponse(
                                name=fc.name,
                                id=fc.id,
                                response={"result": reply}
                            ))
                        # Send local data back to the cloud model
                        await session.send(input={"tool_response": {"function_responses": function_responses}})

            except Exception as e:
                print(f"\n[Receiver Error]: {e}")

        async def sender():
            while self.is_running:
                try:
                    audio_bytes = await asyncio.to_thread(self.mic_queue.get, timeout=0.1)

                    if self.is_playing:
                        # Inject absolute silence to keep connection alive but avoid VAD echo
                        audio_bytes = b'\x00' * len(audio_bytes)

                    await session.send(input={
                        "realtime_input": {
                            "media_chunks": [{
                                "data": audio_bytes,
                                "mime_type": "audio/pcm;rate=16000"
                            }]
                        }
                    })
                except queue.Empty:
                    await asyncio.sleep(0.01)

        await asyncio.gather(sender(), receiver())

    async def run(self):
        if not self.is_running: return
        threading.Thread(target=self.speaker_worker, daemon=True).start()

        tools = [types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="query_local_brain",
                    description="Query the local Gemma 2 9b instance for system admin tasks or bash scripting.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {"prompt": {"type": "STRING", "description": "The command to process locally."}},
                        "required": ["prompt"]
                    }
                ),
                types.FunctionDeclaration(
                    name="read_memory",
                    description="Read the memory.json file to understand your rules and the system hardware."
                ),
                types.FunctionDeclaration(
                    name="update_memory",
                    description="Update the memory.json file.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "category": {"type": "STRING"},
                            "key": {"type": "STRING"},
                            "value": {"type": "STRING"}
                        },
                        "required": ["category", "key", "value"]
                    }
                )
            ]
        )]

        config = {
            "system_instruction": "You are a concise voice assistant. Read your memory file on startup to understand your environment. Keep responses short.",
            "response_modalities": ["AUDIO"],
            "tools": tools,
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}}
        }

        while self.is_running:
            print(f"\nConnecting to {self.model_id}...")
            try:
                async with self.client.aio.live.connect(model=self.model_id, config=config) as session:
                    print("Link established.")
                    with sd.InputStream(samplerate=IN_RATE, channels=CHANNELS, callback=self.audio_callback,
                                      blocksize=CHUNK, dtype='float32', device=self.input_id):
                        await self.main_loop(session)
            except Exception as e:
                print(f"\n[Connection Dropped]: {e}")
                if not self.is_running: break
                await asyncio.sleep(2)

if __name__ == "__main__":
    assistant = HybridLiveAssistant()
    try: asyncio.run(assistant.run())
    except KeyboardInterrupt: pass
    finally: assistant.is_running = False
