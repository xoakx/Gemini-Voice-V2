# 🎤 Gemini Voice Assistant V2

A real-time, low-latency multimodal voice assistant engineered for **Kubuntu 25.10**. This system leverages the **Gemini 2.5 Flash Multimodal Live API** to provide bidirectional audio interaction, system-level automation, and local reasoning capabilities.

---

## 🌟 High-Level Overview
Gemini Voice V2 is designed for stability and "always-on" availability. Unlike previous iterations, V2 focuses on **Session Persistence**—the ability to gracefully recover from WebSocket drops—and **Thread-Safe Hardware Bridging**, ensuring the async event loop remains unblocked by real-time PortAudio callbacks.

### Core Philosophy:
*   **Modular Entry-Points:** Separate modules for production (`pro.py`), hardware testing (`live_audio.py`), and multimodal benchmarking (`core_test.py`).
*   **Hardware Agnostic:** Uses PulseAudio/PipeWire system defaults, allowing seamless routing to Bluetooth (e.g., Soundcore Q20i) or USB peripherals via KDE Sound Settings.
*   **Hybrid Intelligence:** Offloads system-specific queries (hardware specs, bash commands) to a local **Gemma 2 (9b/2b)** instance via Ollama.

---

## ⚙️ Technical Architecture

### Multimodal Live Link
The assistant establishes a full-duplex WebSocket connection to `gemini-2.5-flash-native-audio-latest`. 
*   **Input:** 16kHz Mono PCM Audio.
*   **Output:** 24kHz Mono PCM Audio.
*   **Control Plane:** `asyncio.wait(..., return_when=FIRST_COMPLETED)` logic ensures that if either the sender or receiver task fails, the session is torn down and re-initialized immediately.

### Thread-Safety Bridge
To prevent `RuntimeError` in the async event loop, the PortAudio callback thread utilizes `loop.call_soon_threadsafe()` to inject audio bytes into the `asyncio.Queue`, decoupling high-frequency hardware interrupts from API transmission.

---

## 🛠️ Requirements & Dependencies

### Hardware
*   **CPU:** Intel Ultra 7 265k (optimized for low-latency async processing).
*   **Audio:** Any PulseAudio-compatible input/output device.

### Software
*   **OS:** Kubuntu 25.10
*   **Python:** 3.11+ (Tested on 3.13.7)
*   **Core Libraries:**
    *   `google-genai`: Google's official Multimodal Live SDK.
    *   `sounddevice`: PortAudio interface for Python.
    *   `ollama`: Integration for local model offloading.
    *   `numpy`: Efficient audio buffer manipulation.

---

## 🚀 Installation & Setup

1.  **Clone & Environment:**
    ```bash
    git clone https://github.com/xoakx/Gemini-Voice-V2.git
    cd Gemini-Voice-V2
    python3 -m venv .venv
    source .venv/bin/python3
    pip install -r requirements.txt # Coming soon
    ```

2.  **Configuration:**
    Create a `.env` file in the root:
    ```env
    GEMINI_API_KEY=your_key_here
    ```

3.  **Audio Routing:**
    Verify your device names in `config/audio_config.json`. The assistant targets PulseAudio sinks/sources specifically.

---

## 🏃 Usage

### Production Mode
The most stable version, featuring mic-blocking during AI speech to prevent feedback:
```bash
./.venv/bin/python3 src/modules/pro.py
```

### Full-Duplex Test
For testing simultaneous interruptibility and "Unified Link" stability:
```bash
./.venv/bin/python3 src/modules/live_audio.py
```

### Logic Verification
Run the automated multi-turn tester to verify session persistence:
```bash
./.venv/bin/python3 protest/test_v2_logic.py
```

---

## 🐞 Troubleshooting & Known Issues

*   **"One-Query" Bug:** Resolved in v2.1. Ensure your loop uses `asyncio.wait` instead of `gather`.
*   **1007 Invalid Frame:** Usually caused by silent audio files or incorrect sample rates. The API strictly requires **16kHz Mono PCM**.
*   **Privilege Escalation:** For tasks requiring root (e.g., updating DokuWiki), use the documented **Nested Escalation** method via the `gemmy` account. *Refer to internal system context for credentials.*

---

## 📅 Version History
*   **v2.1 (Current):** Implemented thread-safe loop handling, robust reconnection logic, and 16kHz mono audio validation.
*   **v2.0:** Initial modular refactor.

---

## 🛡️ Security Note
Never commit `.env` or sensitive hardware logs to the repository. The `.gitignore` is pre-configured to exclude environment variables and PortAudio crash dumps.
