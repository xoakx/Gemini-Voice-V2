import os
import asyncio
import wave
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

async def definitive_diagnostic():
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"System: 2.5-Only Diagnostic with Key: {api_key[:8]}...", flush=True)
    
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    
    # 1. Sanity Check - Gemini 2.5
    print("\n[Step 1]: Verifying API Key with Gemini 2.5 Flash...", flush=True)
    try:
        resp = client.models.generate_content(model="models/gemini-2.5-flash", contents="Ping")
        print(f"Result: SUCCESS (Response: {resp.text.strip()})", flush=True)
    except Exception as e:
        print(f"Result: FAILED: {e}", flush=True)

    # 2. Live API Baseline with 2.5 Native Audio
    model_id = "models/gemini-2.5-flash-native-audio-preview-09-2025"
    print(f"\n[Step 2]: Testing Live API with {model_id} on v1alpha...", flush=True)
    
    config = types.LiveConnectConfig(
        system_instruction=types.Content(parts=[types.Part(text="You are a voice assistant. Respond with 'Hello! I am alive!'")]),
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        )
    )
    
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    async def test_session():
        try:
            async with client.aio.live.connect(model=model_id, config=config) as session:
                print("Link Established. Sending boosted audio snippet...", flush=True)
                
                async def receiver():
                    try:
                        async for message in session.receive():
                            print(f"\n[SERVER]: {message}", flush=True)
                    except Exception as e:
                        print(f"Receiver exception: {e}", flush=True)

                async def sender():
                    try:
                        with wave.open("/home/kms/test_input_boost.wav", 'rb') as wf:
                            data = wf.readframes(16000 * 5)
                            chunk_size = 1600
                            for i in range(0, len(data), chunk_size * 2):
                                chunk = data[i:i + chunk_size * 2]
                                await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type='audio/pcm;rate=16000'))
                                await asyncio.sleep(0.1)
                        print("Audio sent. Waiting 20s for processing...", flush=True)
                        await asyncio.sleep(20)
                    except Exception as e:
                        print(f"Sender exception: {e}", flush=True)

                rec_task = asyncio.create_task(receiver())
                send_task = asyncio.create_task(sender())
                await asyncio.wait([rec_task, send_task], return_when=asyncio.FIRST_COMPLETED)
        except Exception as e:
            print(f"Session setup exception: {e}", flush=True)

    try:
        await asyncio.wait_for(test_session(), timeout=35)
    except asyncio.TimeoutError:
        print("\n[Result]: Overall test timed out.", flush=True)

if __name__ == "__main__":
    asyncio.run(definitive_diagnostic())
