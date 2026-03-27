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
        # 1. Services
        if any(word in prompt_lower for word in ["service", "systemctl", "systemd", "failed", "status"]):
            failed_svcs = subprocess.check_output(["systemctl", "--user", "--failed", "--no-pager"], text=True)
            context_data += f"\n--- Failed Services ---\n{failed_svcs}\n"
        # 2. Storage / RAID
        if any(word in prompt_lower for word in ["disk", "drive", "storage", "raid", "mdstat"]):
            lsblk = subprocess.check_output(["lsblk", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE"], text=True)
            context_data += f"\n--- Storage ---\n{lsblk}\n"
            if "raid" in prompt_lower or "mdstat" in prompt_lower:
                mdstat = subprocess.check_output(["cat", "/proc/mdstat"], text=True)
                context_data += f"\n--- RAID ---\n{mdstat}\n"
        # 3. CPU / RAM / Performance
        if any(word in prompt_lower for word in ["cpu", "load", "performance", "temp", "ram", "memory", "free"]):
            uptime = subprocess.check_output(["uptime"], text=True)
            mem = subprocess.check_output(["free", "-h"], text=True)
            context_data += f"\n--- Uptime ---\n{uptime}\n"
            context_data += f"\n--- Memory ---\n{mem}\n"
        # 4. GPU / SRIOV
        if any(word in prompt_lower for word in ["gpu", "sriov", "intel", "vf"]):
            sriov = subprocess.check_output(["bash", "/home/kms/check_sriov_health.sh"], text=True)
            context_data += f"\n--- SRIOV Health ---\n{sriov}\n"
        # 5. Project Context / Memory
        if any(word in prompt_lower for word in ["context", "project", "memory", "history", "log", "summary"]):
            if os.path.exists("/home/kms/GEMINI.md"):
                with open("/home/kms/GEMINI.md", "r") as f:
                    context_data += f"\n--- Project Context (GEMINI.md) ---\n{f.read()}\n"
            # Latest Work Log
            log_dir = "/var/www/html2/dokuwiki/data/pages/work_logs/"
            if os.path.exists(log_dir):
                logs = sorted([f for f in os.listdir(log_dir) if f.endswith(".txt")])
                if logs:
                    with open(os.path.join(log_dir, logs[-1]), "r") as f:
                        context_data += f"\n--- Latest Work Log ({logs[-1]}) ---\n{f.read()}\n"
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
        if not self.is_running: return
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

    async def handle_tool_call(self, session, tool_call):
        responses = []
        for fc in tool_call.function_calls:
            if fc.name == "query_local_brain":
                user_prompt = fc.args.get("prompt")
                print(f"\n[System]: Querying Local Brain (Gemma 2 9b) for: '{user_prompt}'", flush=True)
                try:
                    sys_context = get_system_context(user_prompt)
                    # Augmented prompt for Gemma
                    augmented_prompt = (
                        f"You are a local system administration AI. The user asked: '{user_prompt}'.\n"
                        f"Here is the live system data from the Linux terminal:\n"
                        f"{sys_context}\n"
                        f"Answer the user clearly and concisely based ONLY on this terminal output. "
                        f"Do not read out raw UUIDs. Be helpful but brief."
                    )
                    result = await asyncio.to_thread(ollama.chat, model='gemma2:2b', 
                                                   messages=[{'role': 'user', 'content': augmented_prompt}])
                    
                    responses.append(types.FunctionResponse(
                        name=fc.name, id=fc.id, response={"result": result['message']['content']}
                    ))
                except Exception as e:
                    print(f"Tool Error: {e}", flush=True)
                    responses.append(types.FunctionResponse(name=fc.name, id=fc.id, response={"result": f"Local error: {e}"}))
        
        if responses:
            try:
                await session.send_tool_response(function_responses=responses)
            except Exception as e:
                print(f"Error sending tool response: {e}", flush=True)

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
                        if response.server_content.interrupted:
                            print("\n[System]: Interrupted by user.", flush=True)
                            while not self.speaker_queue.empty():
                                try: self.speaker_queue.get_nowait()
                                except queue.Empty: break
                        self.assistant_speaking = False
                        if response.server_content.turn_complete:
                            print("\n--- Listening ---", flush=True)
                    if response.tool_call:
                        asyncio.create_task(self.handle_tool_call(session, response.tool_call))
            except Exception as e:
                print(f"Receiver Error: {e}", flush=True)

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
            "system_instruction": (
                "You are Gemini Command Pro (v2.1), an expert systems assistant for a high-performance SRIOV environment. NEVER introduce yourself.\n\n"
                "CAPABILITIES:\n"
                "- Memory/RAM: check current usage, free memory, and swap.\n"
                "- CPU: check load average, uptime, and performance.\n"
                "- GPU/SRIOV: check Virtual Function (VF) health and 'VF1 FLR' resets.\n"
                "- Services: check failed systemd units and service status.\n"
                "- Storage: check lsblk layout, RAID status (/proc/mdstat), and mount points.\n"
                "- Project Context: check project history, work logs, and persistent context from GEMINI.md.\n"
                "- Approval Process: aware of the 'Hands-Off' workflow using KDE Connect and '/usr/local/bin/gemini-approve'. Inform user if a CLI command is pending remote approval.\n\n"
                "INSTRUCTIONS:\n"
                "- For any questions about the above, YOU MUST call 'query_local_brain' (powered by Gemma 2 2B).\n"
                "- Use the tool to provide real-time data from the Linux terminal.\n"
                "- Be extremely brief. Start your response immediately."
            ),
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
