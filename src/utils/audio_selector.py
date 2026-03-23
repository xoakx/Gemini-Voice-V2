import json
import os
import subprocess

def get_pulse_devices():
    """Gets a list of all sinks and sources using pactl."""
    sinks = []
    sources = []
    
    try:
        # Get Sinks (Outputs)
        output = subprocess.check_output(["pactl", "list", "sinks"], text=True)
        current_name = ""
        for line in output.split('\n'):
            if "Name: " in line:
                current_name = line.split("Name: ")[1].strip()
            if "Description: " in line and current_name:
                desc = line.split("Description: ")[1].strip()
                sinks.append({"name": current_name, "desc": desc})
                current_name = ""
        
        # Get Sources (Inputs)
        output = subprocess.check_output(["pactl", "list", "sources"], text=True)
        current_name = ""
        for line in output.split('\n'):
            if "Name: " in line:
                current_name = line.split("Name: ")[1].strip()
            if "Description: " in line and current_name:
                desc = line.split("Description: ")[1].strip()
                # Skip monitor sources (loopbacks) for microphones
                if ".monitor" not in current_name:
                    sources.append({"name": current_name, "desc": desc})
                current_name = ""
    except Exception as e:
        print(f"Error fetching PulseAudio devices: {e}")
        
    return sinks, sources

if __name__ == "__main__":
    sinks, sources = get_pulse_devices()
    
    print("\n--- Available Input Devices (Microphones) ---")
    for i, src in enumerate(sources):
        print(f"ID {i}: {src['desc']} ({src['name']})")
        
    print("\n--- Available Output Devices (Speakers/Headphones) ---")
    for i, snk in enumerate(sinks):
        print(f"ID {i}: {snk['desc']} ({snk['name']})")
        
    try:
        in_idx = int(input("\nEnter the ID for your Microphone: "))
        out_idx = int(input("Enter the ID for your Output device: "))
        
        config = {
            "input_pulse_name": sources[in_idx]['name'],
            "output_pulse_name": sinks[out_idx]['name'],
            "input_desc": sources[in_idx]['desc'],
            "output_desc": sinks[out_idx]['desc']
        }
        
        os.makedirs("config", exist_ok=True)
        with open("config/audio_config.json", "w") as f:
            json.dump(config, f, indent=4)
            
        print(f"\n✅ Configuration saved!")
        print(f"Microphone: {config['input_desc']}")
        print(f"Output:     {config['output_desc']}")
        
    except (ValueError, IndexError):
        print("Invalid selection.")
