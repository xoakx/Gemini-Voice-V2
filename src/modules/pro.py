import logging
import os
import sys
import traceback

# SILENCE LOGGING
logging.root.setLevel(logging.CRITICAL)
for name in ["google.genai", "urllib3", "websockets", "absl", "asyncio"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

import json
import asyncio
import sounddevice as sd
import numpy as np
import threading
import queue
import ollama
import subprocess
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Audio Constants - Optimized for Latency
RATE = 48000
CHANNELS = 1
CHUNK = 2048 

def get_system_context(prompt: str) -> str:
    prompt_lower = prompt.lower()
    context_data = ""
    try:
        if any(word in prompt_lower for word in ["service", "systemctl", "systemd", "failed", "status"]):
            failed_svcs = subprocess.check_output(["systemctl", "--user", "--failed", "--no-pager"], text=True)
            context_data += f"\n--- Failed Services ---\n{failed_svcs}\n"
        if any(word in prompt_lower for word in ["disk", "drive", "storage", "raid"]):
            lsblk = subprocess.check_output(["lsblk", "-o", "NAME,SIZE,TYPE,MOUNTPOINT"], text=True)
            context_data += f"\n--- Storage ---\n{lsblk}\n"
        if any(word in prompt_lower for word in ["cpu", "load", "performance", "temp", "ram"]):
            uptime = subprocess.check_output(["uptime"], text=True)
            context_data += f"\n--- Stats ---\n{uptime}\n"
    except Exception: pass
    return context_data

class LiveToolAssistant:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})
        self.mic_queue = queue.Queue(maxsize=50)
        self.speaker_queue = queue.Queue()
        self.is_running = True
        self.assistant_speaking = False
        self.device_name = 'default'

    def audio_callback(self, indata, frames, time_info, status):
        if not self.is_running or self.assistant_speaking: return
        try:
            # Resample 48k -> 16k
            resampled = indata[::3].flatten()
            audio_int16 = (resampled * 32767).astype(np.int16).tobytes()
            self.mic_queue.put_nowait(audio_int16)
        except Exception: pass

    def speaker_worker(self):
        print("DEBUG: Speaker Worker Active", flush=True)
        try:
            with sd.OutputStream(samplerate=RATE, channels=CHANNELS, dtype='int16', device=self.device_name, blocksize=CHUNK) as stream:
                while self.is_running:
                    try:
                        data_24k = np.frombuffer(self.speaker_queue.get(timeout=0.1), dtype='int16')
                        # Resample 24k -> 48k (assuming model sends 24kHz)
                        data_48k = np.repeat(data_24k, 2)
                        stream.write(data_48k)
                    except queue.Empty: continue
                    except Exception: pass
        except Exception as e:
            print(f"CRITICAL: Output Hardware Error: {e}", flush=True)

    async def main_loop(self, session):
        async def receiver():
            try:
                async for response in session.receive():
                    if not self.is_running: break
                    if response.server_content and response.server_content.model_turn:
                        self.assistant_speaking = True
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data:
                                self.speaker_queue.put(part.inline_data.data)
                            if part.text:
                                print(f"\n[Brain]: {part.text}", flush=True)
                    if response.server_content and (response.server_content.interrupted or response.server_content.turn_complete):
                        self.assistant_speaking = False
                        if response.server_content.turn_complete:
                            print("\n--- Listening ---", flush=True)
                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            if fc.name == "query_local_brain":
                                sys_context = get_system_context(fc.args.get("prompt"))
                                result = await asyncio.to_thread(ollama.chat, model='gemma2:9b', messages=[{'role': 'user', 'content': f"Context:\n{sys_context}\nAnswer concisely."}])
                                await session.send_tool_response(function_responses=[types.FunctionResponse(name=fc.name, id=fc.id, response={"result": result['message']['content']})])
            except Exception: pass

        async def sender():
            try:
                while self.is_running:
                    try:
                        audio_bytes = await asyncio.to_thread(self.mic_queue.get, timeout=0.1)
                        await session.send_realtime_input(audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000"))
                        await asyncio.sleep(0)
                    except queue.Empty: await asyncio.sleep(0.01)
            except Exception: pass

        await asyncio.wait([asyncio.create_task(sender()), asyncio.create_task(receiver())], return_when=asyncio.FIRST_COMPLETED)

    async def run(self):
        threading.Thread(target=self.speaker_worker, daemon=True).start()
        local_brain_tool = types.Tool(function_declarations=[types.FunctionDeclaration(
            name="query_local_brain",
            description="Query local system status.",
            parameters={"type": "OBJECT", "properties": {"prompt": {"type": "STRING"}}, "required": ["prompt"]}
        )])
        
        config = {
            "system_instruction": "You are Gemini Command Pro (v2.1), an expert systems assistant for a Kubuntu 25.10 / Windows 11 SRIOV environment. NEVER introduce yourself. DO NOT say 'I am Gemini' or 'Gemini Command Pro'.\n\nSYSTEM CONTEXT:\n- Host: Intel Ultra 7 265K (Arrow Lake), 64GB DDR5, SRIOV iGPU (i915-sriov-dkms).\n- VM: 'rsl-primary' (Windows 11) uses VF 00:02.1 for Raid Shadow Legends.\n- Approval: Remote workflow enabled via KDE Connect and '/usr/local/bin/gemini-approve'. Inform user if commands are pending remote approval.\n- Monitoring: Watch for 'VF1 FLR' resets and thermal throttling.\n\nCOMMUNICATION:\n- Be extremely brief and direct. Start your response immediately with the answer.",
            "response_modalities": ["AUDIO"],
            "tools": [local_brain_tool],
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}}
        }
        
        while self.is_running:
            print(f"\n🚀 Launching Smooth Link...", flush=True)
            try:
                async with self.client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=config) as session:
                    print("✅ LINK STABLE.", flush=True)
                    with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=self.audio_callback, blocksize=CHUNK, dtype='float32', device=self.device_name):
                        await self.main_loop(session)
            except Exception as e:
                print(f"CRITICAL failure:\n{traceback.format_exc()}", flush=True)
                await asyncio.sleep(2)

if __name__ == "__main__":
    assistant = LiveToolAssistant()
    try: asyncio.run(assistant.run())
    except KeyboardInterrupt: pass
    finally: assistant.is_running = False
