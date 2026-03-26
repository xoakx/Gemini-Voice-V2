import os
import sys
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

# Audio Constants
IN_RATE = 16000
CHANNELS = 1
CHUNK = 4096 

def get_system_context(prompt: str) -> str:
    """Runs local bash commands based on keywords in the prompt."""
    prompt_lower = prompt.lower()
    context_data = ""

    # 1. Systemd / Service queries
    if any(word in prompt_lower for word in ["service", "systemctl", "systemd", "failed", "status"]):
        try:
            failed_svcs = subprocess.check_output(["systemctl", "--failed", "--no-pager"], text=True)
            context_data += f"\n--- Systemd Failed Services ---\n{failed_svcs}\n"
        except Exception as e:
            context_data += f"\nError checking systemctl: {e}\n"

    # 2. Disk / Drive / Storage / RAID
    if any(word in prompt_lower for word in ["disk", "drive", "storage", "m.2", "luks", "raid", "mdstat"]):
        try:
            lsblk_out = subprocess.check_output(["lsblk", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE"], text=True)
            context_data += f"\n--- Storage Layout (lsblk) ---\n{lsblk_out}\n"
            if "raid" in prompt_lower or "mdstat" in prompt_lower:
                mdstat = subprocess.check_output(["cat", "/proc/mdstat"], text=True)
                context_data += f"\n--- RAID Status (/proc/mdstat) ---\n{mdstat}\n"
        except Exception as e:
            context_data += f"\nError checking storage: {e}\n"

    # 3. Performance / CPU / Temp / RAM
    if any(word in prompt_lower for word in ["cpu", "load", "performance", "temp", "ram", "memory", "free", "hot"]):
        try:
            uptime = subprocess.check_output(["uptime"], text=True)
            mem = subprocess.check_output(["free", "-h"], text=True)
            sensors = subprocess.check_output(["sensors"], text=True)
            context_data += f"\n--- CPU Load & Uptime ---\n{uptime}\n"
            context_data += f"\n--- Memory Usage ---\n{mem}\n"
            context_data += f"\n--- Hardware Sensors ---\n{sensors}\n"
        except Exception as e:
            context_data += f"\nError checking performance: {e}\n"

    # 4. Virtual Machine (KVM/QEMU)
    if any(word in prompt_lower for word in ["vm", "virtual machine", "qemu", "kvm", "windows"]):
        try:
            virsh_out = subprocess.check_output(["virsh", "list", "--all"], text=True)
            context_data += f"\n--- KVM/QEMU Virtual Machines ---\n{virsh_out}\n"
        except Exception as e:
            pass 

    # 5. GPU / SRIOV / Graphics
    if any(word in prompt_lower for word in ["gpu", "intel", "sriov", "graphics", "video"]):
        try:
            sriov_health = subprocess.check_output(["bash", "/home/kms/check_sriov_health.sh"], text=True)
            context_data += f"\n--- SRIOV iGPU Health ---\n{sriov_health}\n"
        except Exception as e:
            pass

    # 6. Network status
    if any(word in prompt_lower for word in ["network", "ip", "internet", "online"]):
        try:
            ip_addr = subprocess.check_output(["ip", "-br", "addr"], text=True)
            context_data += f"\n--- Network Interfaces ---\n{ip_addr}\n"
        except Exception as e:
            pass

    # 7. Kernel Errors / Dmesg
    if any(word in prompt_lower for word in ["error", "kernel", "crash", "dmesg", "logs"]):
        try:
            dmesg = subprocess.check_output(["dmesg", "--tail", "15"], text=True)
            context_data += f"\n--- Recent Kernel Logs ---\n{dmesg}\n"
        except Exception as e:
            pass

    # 8. Btrfs / Filesystem / Scrub
    if any(word in prompt_lower for word in ["btrfs", "scrub", "filesystem", "health"]):
        try:
            # Dynamically get all Btrfs mount points
            mounts = subprocess.check_output(["bash", "-c", "df -hT | grep btrfs | awk '{print $7}'"], text=True).splitlines()
            for m in mounts:
                stats = subprocess.check_output(["bash", "-c", f"echo iscrewedupdaputer | sudo -S btrfs device stats {m}"], text=True)
                scrub = subprocess.check_output(["bash", "-c", f"echo iscrewedupdaputer | sudo -S btrfs scrub status {m}"], text=True)
                context_data += f"\n--- Btrfs Health: {m} ---\n{stats}\n{scrub}\n"
        except Exception as e:
            pass

    return context_data

class TextOnlyAssistant:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_version = 'v1alpha'
        self.model_id = "gemini-2.5-flash-native-audio-latest"
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': self.api_version})

        self.mic_queue = queue.Queue(maxsize=20)
        self.is_running = True
        self._load_config()

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "../../config/audio_config.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                devices = sd.query_devices()
                self.input_id = next((i for i, d in enumerate(devices) if config["input_pulse_name"].lower() in d['name'].lower() and d['max_input_channels'] > 0), None)
        except Exception as e:
            print(f"Config error: {e}. Using system default input.")
            self.input_id = None

    def audio_callback(self, indata, frames, time_info, status):
        """Mic -> Queue"""
        if not self.is_running: return
        audio_int16 = (indata * 32767).astype(np.int16).tobytes()
        try:
            self.mic_queue.put_nowait(audio_int16)
        except queue.Full: pass

    async def main_loop(self, session):
        print("\n[System]: Link Established. (Multi-turn Mode)")
        print("[System]: Assistant audio is discarded; only TEXT is displayed.\n")

        async def receiver():
            try:
                async for response in session.receive():
                    if not self.is_running: break

                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.text:
                                print(f"{part.text}", end="", flush=True)

                    if response.server_content and response.server_content.turn_complete:
                        print("\n\n--- Listening ---")

                    if response.tool_call:
                        function_responses = []
                        for fc in response.tool_call.function_calls:
                            if fc.name == "query_local_brain":
                                user_prompt = fc.args.get("prompt")
                                print(f"\n[System]: Querying Local Brain for: '{user_prompt}'")
                                
                                # 1. Grab the live system state
                                sys_context = get_system_context(user_prompt)
                                
                                # 2. Build the augmented prompt for Gemma
                                augmented_prompt = (
                                    f"You are a local system administration AI. The user asked: '{user_prompt}'.\n"
                                    f"Here is the live system data from the Linux terminal:\n"
                                    f"{sys_context}\n"
                                    f"Answer the user clearly and concisely based ONLY on this terminal output. "
                                    f"Do not read out raw UUIDs or exact byte sizes; summarize it naturally."
                                )

                                try:
                                    # 3. Send the augmented prompt to Gemma
                                    result = await asyncio.to_thread(
                                        ollama.chat, 
                                        model='gemma2:2b', 
                                        messages=[{'role': 'user', 'content': augmented_prompt}]
                                    )
                                    
                                    # 4. Return Gemma's summarized answer back to Gemini
                                    function_responses.append(
                                        types.FunctionResponse(
                                            name=fc.name, 
                                            id=fc.id, 
                                            response={"result": result['message']['content']}
                                        )
                                    )
                                except Exception as e:
                                    function_responses.append(
                                        types.FunctionResponse(name=fc.name, id=fc.id, response={"result": f"Local error: {e}"})
                                    )
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
                    except queue.Empty:
                        await asyncio.sleep(0.01)
            except Exception as e:
                print(f"\n[Sender Error]: {e}")

        # Robust Wait logic from pro.py
        done, pending = await asyncio.wait(
            [asyncio.create_task(sender()), asyncio.create_task(receiver())],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        print("\n[System]: Cycle Resetting...")

    async def run(self):
        local_brain_tool = types.Tool(function_declarations=[types.FunctionDeclaration(
            name="query_local_brain",
            description="Use local Gemma 2 2b for system admin and hardware status.",
            parameters={"type": "OBJECT", "properties": {"prompt": {"type": "STRING"}}, "required": ["prompt"]}
        )])

        config = {
            "system_instruction": "You are a concise assistant. Respond via text and audio.",
            "response_modalities": ["AUDIO"],
            "tools": [local_brain_tool],
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}}
        }

        while self.is_running:
            try:
                async with self.client.aio.live.connect(model=self.model_id, config=config) as session:
                    with sd.InputStream(samplerate=IN_RATE, channels=CHANNELS, callback=self.audio_callback,
                                      blocksize=CHUNK, dtype='float32', device=self.input_id):
                        await self.main_loop(session)
            except Exception as e:
                print(f"\n[Connection Drop]: {e}")
                await asyncio.sleep(2)

if __name__ == "__main__":
    assistant = TextOnlyAssistant()
    try: asyncio.run(assistant.run())
    except KeyboardInterrupt: pass
    finally: assistant.is_running = False
