import os
import json
import base64
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def print_serialized_json():
    client = genai.Client(api_key="TEST")
    # Simulate a realtime input
    audio_blob = types.Blob(data=b"test_data", mime_type="audio/pcm;rate=16000")
    
    # We can't easily call the private converters, but we can look at what LiveSendRealtimeInputParameters expects
    params = types.LiveSendRealtimeInputParameters(audio=audio_blob)
    print(f"Params Model: {params.model_dump(exclude_none=True)}")
    
    # Let's try to find where it's actually converted
    # In live.py: realtime_input_dict = live_converters._LiveSendRealtimeInputParameters_to_mldev(from_object=realtime_input)
    
if __name__ == "__main__":
    print_serialized_json()
