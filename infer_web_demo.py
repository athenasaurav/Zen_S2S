import socketio
import time
import json
import numpy as np
import wave
import sys

# ========== USER CONFIGURATION =============
SERVER_URL = "http://172.17.0.2:1000"  # Change to your server's address
AUDIO_FILE = "./sample_001.wav"      # Path to your input audio file
OUTPUT_AUDIO_FILE = "output_infer.wav"   # Where to save the output audio
SAMPLE_RATE = 16000                      # Match your server's expected rate
# ===========================================

# State
first_audio_time = None
first_text_time = None
start_time = None
received_audio = []
received_text = []

# SocketIO client
sio = socketio.Client()

@sio.event
def connect():
    print("Connected to server")
    global start_time
    start_time = time.time()
    # Send audio after connect
    with wave.open(AUDIO_FILE, 'rb') as wf:
        audio_data = wf.readframes(wf.getnframes())
        # Convert to int16 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        # The server expects int16 bytes, so send as is
        payload = {
            "audio": list(audio_data),  # Send as list of ints for JSON serialization
            "sample_rate": SAMPLE_RATE
        }
        sio.emit("audio", json.dumps(payload))
        print("Audio sent")

@sio.on("audio")
def on_audio(data):
    global first_audio_time
    if first_audio_time is None:
        first_audio_time = time.time()
        print(f"TTFB (audio): {first_audio_time - start_time:.3f} seconds")
    # Save audio bytes
    received_audio.append(data)
    print(f"Received audio chunk: {len(data)} bytes")

@sio.on("first_audio_time")
def on_first_audio_time(data):
    print("Server-side first audio time:", data)

@sio.on("text")
def on_text(data):
    global first_text_time
    if first_text_time is None:
        first_text_time = time.time()
        print(f"TTFT (text): {first_text_time - start_time:.3f} seconds")
    received_text.append(data)
    print("Received text:", data)

@sio.on("audio_end")
def on_audio_end(data):
    print("Received end of audio signal from server.")
    sio.disconnect()

@sio.event
def disconnect():
    print("Disconnected from server")
    # Save audio to file
    if received_audio:
        audio_bytes = b"".join(received_audio)
        with wave.open(OUTPUT_AUDIO_FILE, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        print(f"Saved output audio to {OUTPUT_AUDIO_FILE}")
    if received_text:
        print("Full text received:", "".join(received_text))

if __name__ == "__main__":
    try:
        sio.connect(SERVER_URL, transports=["websocket"], socketio_path="/socket.io")
        sio.wait()
    except Exception as e:
        print(f"Error connecting or running inference: {e}")
        sys.exit(1) 