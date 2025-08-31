

import copy
import math
import os
import sys
import time
import warnings
import re
import queue
import threading

from datetime import datetime, timezone

import gradio as gr
import numpy as np
import torch
from numba import jit
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Add GLM-4-Voice paths (ENABLE GLM-4-Voice tokenizer)
if True:  # Changed from False to True!
    # glm4voice tokenizer
    sys.path.append("third_party/GLM-4-Voice/")
    sys.path.append("third_party/GLM-4-Voice/cosyvoice/")
    sys.path.append("third_party/GLM-4-Voice/third_party/Matcha-TTS/")

    # Updated paths to match your setup
    audio_tokenizer_path = "./models/THUDM/glm-4-voice-tokenizer"
    flow_path = "./models/THUDM/glm-4-voice-decoder"
    audio_tokenizer_type = "glm4voice"
    model_name_or_path = "./models/VITA-MLLM/VITA-Audio-Boost"

# Import VITA-Audio modules
try:
    from vita_audio.data.processor.audio_processor import add_audio_input_contiguous
    from vita_audio.tokenizer import get_audio_tokenizer
    AUDIO_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Audio modules not available: {e}")
    AUDIO_MODULES_AVAILABLE = False

PUNCTUATION = "!?.,;:~…@#$%^&*()_+-=[]{}|\\`\"'<>/\n\t "

def get_utc_timestamp():
    """Get current UTC timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()

def get_monotonic_time():
    """Get monotonic time for precise latency measurements"""
    return time.monotonic()

def log_event(event_type, task_type, message, **kwargs):
    """Log event with UTC timestamp and task context"""
    timestamp = get_utc_timestamp()
    task_info = f"[{task_type}]" if task_type else ""
    print(f"{timestamp} {task_info} {event_type}: {message}")
    
    # Log additional kwargs if provided
    for key, value in kwargs.items():
        if value is None:
            continue
        print(f"{timestamp} {task_info} {event_type}: {key}: {value}")

def log_latency_metric(metric_name, start_time, end_time, task_type="LATENCY", **kwargs):
    """Log latency metric with precise timing"""
    latency_ms = (end_time - start_time) * 1000
    latency_s = end_time - start_time
    
    log_event(f"LATENCY_{metric_name}", task_type, f"{metric_name}: {latency_ms:.2f}ms ({latency_s:.6f}s)", 
              latency_ms=f"{latency_ms:.2f}",
              latency_s=f"{latency_s:.6f}",
              start_time=f"{start_time:.6f}",
              end_time=f"{end_time:.6f}",
              **kwargs)

@jit
def wav_to_int16(audio: np.ndarray) -> np.ndarray:
    am = int(math.ceil(float(np.abs(audio).max())) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)

def is_wav(file_path):
    if file_path is None:
        return False
    wav_extensions = [".wav", ".mp3", ".flac", ".ogg"]
    _, ext = os.path.splitext(file_path)
    return ext.lower() in wav_extensions

def clean_text_display(text, task_type="Spoken QA"):
    """Enhanced text cleaning to remove system message artifacts and audio tokens"""
    
    log_event("CLEAN", task_type, f"Cleaning text for {task_type}")
    log_event("CLEAN", task_type, f"Original text: {text[:200]}...")
    
    # Remove system/user/assistant markers
    clean_text = text
    clean_text = clean_text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    clean_text = re.sub(r"(system|user|assistant)\s*", "", clean_text)
    
    # Remove ALL audio-related tokens for clean display
    # Remove audio segments completely
    clean_text = re.sub(r"<\|begin_of_audio\|>.*?<\|end_of_audio\|>", "", clean_text, flags=re.DOTALL)
    
    # Remove any remaining audio tokens
    clean_text = re.sub(r"<\|audio_\d+\|>", "", clean_text)
    
    # Remove standalone <|audio|> tokens
    clean_text = clean_text.replace("<|audio|>", "")
    
    # ENHANCED: Remove system message artifacts more aggressively
    system_artifacts = [
        "You are a helpful AI assistant.",
        "You are a helpful AI assistant",
        "You are a helpful AI .",
        "You are a helpful AI.",
        "You are a helpful AI",
        "You are a helpful AI . .",  # With extra spaces/dots
        "You are a helpful AI  .",   # With double spaces
        "You are a helpful AI   .",  # With triple spaces
    ]
    
    for artifact in system_artifacts:
        clean_text = clean_text.replace(artifact, "")
    
    # Clean up extra whitespace and newlines
    clean_text = re.sub(r"\n\s*\n", "\n", clean_text)  # Remove multiple newlines
    clean_text = re.sub(r"^\s+|\s+$", "", clean_text)  # Remove leading/trailing whitespace
    clean_text = re.sub(r"\s+", " ", clean_text)  # Normalize spaces
    
    # Remove leading dots, periods, or punctuation
    clean_text = re.sub(r"^[.\s,;:!?]+", "", clean_text)
    
    # Final cleanup: remove any remaining leading/trailing whitespace and dots
    clean_text = clean_text.strip()
    while clean_text.startswith(".") or clean_text.startswith(" "):
        clean_text = clean_text[1:].strip()
    
    final_text = clean_text.strip()
    log_event("CLEAN", task_type, f"Final cleaned text: '{final_text}'")
    
    return final_text

class RealTimeAudioProcessor:
    """Real-time audio processor that decodes audio tokens immediately as they appear"""
    
    def __init__(self, audio_tokenizer, prompt_audio_path=None):
        self.audio_tokenizer = audio_tokenizer
        self.prompt_audio_path = prompt_audio_path
        self.audio_segments = []  # Store individual audio segments
        self.audio_files = []     # Store saved audio files
        self.segment_counter = 0
        
        # Create output directory
        os.makedirs("/tmp/realtime_audio_segments", exist_ok=True)
        
        log_event("INIT", "REALTIME_AUDIO", "Real-time audio processor initialized")
    
    def process_audio_tokens_immediate(self, audio_tokens, segment_id):
        """Process audio tokens immediately and generate audio segment"""
        if not self.audio_tokenizer or not audio_tokens:
            return None
            
        try:
            decode_start_time = time.time()
            
            log_event("REALTIME_DECODE", "AUDIO", f"Starting immediate decode for segment {segment_id}", 
                      token_count=len(audio_tokens),
                      first_token=audio_tokens[0] if audio_tokens else None,
                      last_token=audio_tokens[-1] if audio_tokens else None)
            
            # Decode audio tokens immediately
            audio_segment = self.audio_tokenizer.decode(
                audio_tokens, 
                source_speech_16k=self.prompt_audio_path
            )
            
            decode_time = time.time() - decode_start_time
            
            if audio_segment is not None:
                # Save audio segment
                audio_file = f"/tmp/realtime_audio_segments/segment_{segment_id}_{int(time.time()*1000)}.wav"
                
                try:
                    import soundfile as sf
                    sf.write(audio_file, audio_segment, 16000)
                    
                    self.audio_segments.append(audio_segment)
                    self.audio_files.append(audio_file)
                    
                    log_event("REALTIME_DECODE", "AUDIO", f"Audio segment {segment_id} generated successfully", 
                              decode_time_ms=f"{decode_time*1000:.1f}",
                              audio_shape=audio_segment.shape,
                              audio_length_s=f"{audio_segment.shape[0]/16000:.2f}",
                              file=os.path.basename(audio_file))
                    
                    return audio_file
                    
                except Exception as e:
                    log_event("REALTIME_DECODE", "AUDIO", f"Error saving audio segment {segment_id}: {e}")
                    return None
            else:
                log_event("REALTIME_DECODE", "AUDIO", f"Failed to decode audio segment {segment_id}")
                return None
                
        except Exception as e:
            log_event("REALTIME_DECODE", "AUDIO", f"Error processing audio segment {segment_id}: {e}")
            return None
    
    def get_latest_audio_segment(self):
        """Get the latest audio segment for immediate playback"""
        return self.audio_files[-1] if self.audio_files else None
    
    def get_all_audio_segments(self):
        """Get all generated audio segments"""
        return self.audio_files
    
    def get_segment_count(self):
        """Get total number of audio segments generated"""
        return len(self.audio_segments)

class StreamingS2SInference:
    """Streaming Speech-to-Speech Inference with REAL-TIME audio decoding"""
    
    def __init__(self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, flow_path, audio_tokenizer_rank=0):
        self.model_name_or_path = model_name_or_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.audio_tokenizer_type = audio_tokenizer_type
        self.flow_path = flow_path
        self.audio_tokenizer_rank = audio_tokenizer_rank
        
        # Load tokenizer
        log_event("INIT", "SYSTEM", "Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        log_event("INIT", "SYSTEM", "Tokenizer loaded")
        
        # Load model
        log_event("INIT", "SYSTEM", "Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        log_event("INIT", "SYSTEM", "Model loaded")
        
        # Load audio tokenizer
        if AUDIO_MODULES_AVAILABLE:
            try:
                self.audio_tokenizer = get_audio_tokenizer(
                    audio_tokenizer_path,
                    audio_tokenizer_type,
                    flow_path=flow_path,
                    rank=audio_tokenizer_rank,
                )
                log_event("INIT", "SYSTEM", "GLM-4-Voice audio tokenizer loaded")
            except Exception as e:
                log_event("INIT", "SYSTEM", f"Error loading audio tokenizer: {e}")
                self.audio_tokenizer = None
        else:
            self.audio_tokenizer = None
            
        # Configure generation
        self.model.generation_config.max_new_tokens = 8192
        self.model.generation_config.chat_format = "chatml"
        self.model.generation_config.max_window_size = 8192
        self.model.generation_config.use_cache = True
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_k = 50
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.num_beams = 1
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        if self.model.config.model_type == "hunyuan":
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
            
        # Default system message
        self.default_system_message = [
            {
                "role": "system", 
                "content": "You are a helpful AI assistant.",
            }
        ]
        
        # Get audio offset
        self.audio_offset = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")
        log_event("INIT", "SYSTEM", f"Audio offset: {self.audio_offset}")

    def stream_spoken_qa_realtime(self, audio_path, max_returned_tokens=2048):
        """Stream Spoken QA with REAL-TIME audio decoding as tokens appear"""
        
        # Start timing
        request_start_time = time.time()
        log_event("STREAM_START", "Spoken QA", "Starting real-time streaming with immediate audio decoding", 
                  audio_file=os.path.basename(audio_path))
        
        # Prepare messages for Spoken QA
        messages = self.default_system_message + [
            {
                "role": "user", 
                "content": "<|audio|>",
            }
        ]
        
        # Apply chat template
        template_start_time = time.time()
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        template_time = time.time() - template_start_time
        log_event("TEMPLATE", "Spoken QA", "Chat template applied", 
                  input_tokens=input_ids.shape[1], 
                  template_time_ms=f"{template_time*1000:.1f}")

        # Process audio input
        audio_encode_start_time = time.time()
        if self.audio_tokenizer and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            log_event("AUDIO_ENCODING_START", "Spoken QA", "Starting audio encoding")
            
            # Encode audio to tokens
            audio_tokens = self.audio_tokenizer.encode(audio_path)
            encode_time = time.time() - audio_encode_start_time
            audio_tokens_str = "".join([f"<|audio_{i}|>" for i in audio_tokens])
            
            log_event("AUDIO_ENCODING_COMPLETE", "Spoken QA", "Audio encoded to tokens", 
                      audio_tokens=len(audio_tokens), 
                      encode_time_ms=f"{encode_time*1000:.1f}")
            
            # Replace <|audio|> in the input with actual audio tokens
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            input_text = input_text.replace("<|audio|>", f"<|begin_of_audio|>{audio_tokens_str}<|end_of_audio|>")
            
            # Re-tokenize with audio tokens
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        else:
            encode_time = time.time() - audio_encode_start_time
            log_event("AUDIO_ENCODING_SKIPPED", "Spoken QA", "Audio encoding skipped", 
                      encode_time_ms=f"{encode_time*1000:.1f}")

        # Move to device
        input_ids = input_ids.to(self.model.device)

        # Initialize streaming components
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            timeout=30.0, 
            skip_prompt=True, 
            skip_special_tokens=False
        )
        
        # Initialize real-time audio processor
        realtime_audio = RealTimeAudioProcessor(self.audio_tokenizer, audio_path)
        
        # Generation parameters
        generation_kwargs = {
            "input_ids": input_ids,
            "num_logits_to_keep": 1,
            "max_new_tokens": max_returned_tokens,
            "use_cache": True,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        log_event("GENERATION_START", "Spoken QA", "Starting real-time streaming generation with immediate audio decoding")
        generation_start_time = time.time()
        
        # Start generation in background thread
        generation_thread = threading.Thread(
            target=self.model.generate, 
            kwargs=generation_kwargs
        )
        generation_thread.start()
        
        # Stream processing variables
        full_response = ""
        current_text_buffer = ""
        current_audio_tokens = []
        first_token_time = None
        first_text_token_time = None
        first_audio_token_time = None
        token_count = 0
        segment_counter = 0
        
        # Track audio segment boundaries
        in_audio_segment = False
        current_segment_tokens = []
        
        log_event("STREAM_PROCESSING", "Spoken QA", "Starting real-time token streaming with IMMEDIATE audio decoding")
        
        # Process tokens as they arrive
        for new_token in streamer:
            current_time = time.time()
            token_count += 1
            
            # Track first token (any type)
            if first_token_time is None:
                first_token_time = current_time
                ttft = (first_token_time - generation_start_time) * 1000
                ttft_from_server = (first_token_time - request_start_time) * 1000
                log_event("TTFT", "Spoken QA", f"First token generated", 
                          token=new_token.strip()[:50], 
                          ttft_ms=f"{ttft:.1f}",
                          ttft_from_server_ms=f"{ttft_from_server:.1f}")
            
            full_response += new_token
            
            # Check for audio segment boundaries
            if "<|begin_of_audio|>" in new_token:
                in_audio_segment = True
                current_segment_tokens = []
                log_event("AUDIO_SEGMENT", "Spoken QA", f"🎵 Audio segment {segment_counter + 1} started")
                
            elif "<|end_of_audio|>" in new_token:
                in_audio_segment = False
                segment_counter += 1
                
                # Track first audio token timing
                if first_audio_token_time is None:
                    first_audio_token_time = current_time
                    ttfat = (first_audio_token_time - generation_start_time) * 1000
                    ttfat_from_server = (first_audio_token_time - request_start_time) * 1000
                    log_event("TTFAT", "Spoken QA", f"First audio segment completed", 
                              ttfat_ms=f"{ttfat:.1f}",
                              ttfat_from_server_ms=f"{ttfat_from_server:.1f}")
                
                # IMMEDIATE AUDIO DECODING - Process this segment right now!
                if current_segment_tokens:
                    log_event("REALTIME_DECODE", "Spoken QA", f"🎯 IMMEDIATE decoding of segment {segment_counter} with {len(current_segment_tokens)} tokens")
                    
                    # Start audio decoding in background thread to avoid blocking
                    decode_thread = threading.Thread(
                        target=realtime_audio.process_audio_tokens_immediate,
                        args=(current_segment_tokens, segment_counter)
                    )
                    decode_thread.start()
                    
                    # Get the latest audio segment for immediate playback
                    latest_audio = realtime_audio.get_latest_audio_segment()
                    
                    # Yield streaming update with immediate audio
                    yield {
                        'type': 'audio_segment_complete',
                        'segment_id': segment_counter,
                        'text': clean_text_display(full_response),
                        'audio_tokens_count': len(current_segment_tokens),
                        'audio_file': latest_audio,
                        'total_tokens': token_count,
                        'ttft_ms': (first_token_time - generation_start_time) * 1000 if first_token_time else None,
                        'ttft_from_server_ms': (first_token_time - request_start_time) * 1000 if first_token_time else None,
                        'ttfat_ms': (first_audio_token_time - generation_start_time) * 1000 if first_audio_token_time else None,
                        'audio_encode_time_ms': encode_time * 1000,
                        'segment_count': segment_counter
                    }
                
                current_segment_tokens = []
                
            elif in_audio_segment:
                # Extract audio token IDs from this token
                audio_tokens_in_token = re.findall(r"<\|audio_(\d+)\|>", new_token)
                for audio_token_id in audio_tokens_in_token:
                    token_id = int(audio_token_id) - self.audio_offset
                    current_segment_tokens.append(token_id)
                    current_audio_tokens.append(token_id)
                    
                    log_event("AUDIO_TOKEN_STREAMING", "Spoken QA", f"Audio token collected for segment {segment_counter + 1}", 
                              token_id=token_id,
                              segment_tokens=len(current_segment_tokens))
            else:
                # This is a text token
                if first_text_token_time is None and new_token.strip() and not new_token.startswith('<|'):
                    first_text_token_time = current_time
                    ttft_text = (first_text_token_time - generation_start_time) * 1000
                    ttft_text_from_server = (first_text_token_time - request_start_time) * 1000
                    log_event("TTFT_TEXT", "Spoken QA", f"First text token generated", 
                              ttft_text_ms=f"{ttft_text:.1f}",
                              ttft_text_from_server_ms=f"{ttft_text_from_server:.1f}",
                              token=new_token.strip()[:50])
                
                current_text_buffer += new_token
                
                # Yield streaming update with text
                yield {
                    'type': 'text_update',
                    'text': clean_text_display(full_response),
                    'total_tokens': token_count,
                    'audio_tokens_count': len(current_audio_tokens),
                    'ttft_ms': (first_token_time - generation_start_time) * 1000 if first_token_time else None,
                    'ttft_text_ms': (first_text_token_time - generation_start_time) * 1000 if first_text_token_time else None,
                    'ttft_from_server_ms': (first_token_time - request_start_time) * 1000 if first_token_time else None,
                    'audio_encode_time_ms': encode_time * 1000,
                    'segment_count': segment_counter
                }
        
        # Wait for generation to complete
        generation_thread.join()
        generation_end_time = time.time()
        
        # Wait for any remaining audio decoding to complete
        time.sleep(1.0)  # Give background decode threads time to finish
        
        total_generation_time = generation_end_time - generation_start_time
        total_inference_time = generation_end_time - request_start_time
        
        log_event("GENERATION_COMPLETE", "Spoken QA", "Real-time streaming generation completed", 
                  total_time_s=f"{total_generation_time:.2f}",
                  total_time_from_server_s=f"{total_inference_time:.2f}",
                  total_tokens=token_count,
                  audio_tokens=len(current_audio_tokens),
                  segments_generated=segment_counter)
        
        # Final result with all audio segments
        yield {
            'type': 'final_result',
            'full_text': clean_text_display(full_response),
            'audio_files': realtime_audio.get_all_audio_segments(),
            'total_tokens': token_count,
            'generation_time_s': total_generation_time,
            'total_time_s': total_inference_time,
            'audio_encode_time_ms': encode_time * 1000,
            'ttft_ms': (first_token_time - generation_start_time) * 1000 if first_token_time else None,
            'ttft_text_ms': (first_text_token_time - generation_start_time) * 1000 if first_text_token_time else None,
            'ttfat_ms': (first_audio_token_time - generation_start_time) * 1000 if first_audio_token_time else None,
            'ttft_from_server_ms': (first_token_time - request_start_time) * 1000 if first_token_time else None,
            'segment_count': segment_counter
        }

def create_realtime_streaming_interface(s2s_engine):
    """Create real-time streaming interface with immediate audio decoding"""
    
    def process_audio_realtime_streaming(audio_file, progress=gr.Progress()):
        """Process audio with real-time streaming and immediate audio decoding"""
        
        if audio_file is None:
            return "❌ Please upload or record an audio file", "", None, "No audio provided", None, 0
        
        log_event("UI_REQUEST", "Spoken QA", "Starting real-time streaming request with immediate audio decoding", 
                  audio_file=os.path.basename(audio_file))
        
        # Initialize display variables
        current_text = ""
        status_text = "🎯 Processing audio input with real-time audio decoding..."
        audio_output = None
        latest_audio_segment = None
        segment_count = 0
        
        try:
            # Stream the inference
            for update in s2s_engine.stream_spoken_qa_realtime(audio_file):
                
                if update['type'] == 'text_update':
                    current_text = update['text']
                    segment_count = update.get('segment_count', 0)
                    
                    # Build status with timing metrics
                    status_parts = [f"🔄 Streaming... {update['total_tokens']} tokens"]
                    
                    if update.get('ttft_from_server_ms'):
                        status_parts.append(f"TTFT-Server: {update['ttft_from_server_ms']:.1f}ms")
                    if update['audio_encode_time_ms']:
                        status_parts.append(f"Audio-Encode: {update['audio_encode_time_ms']:.1f}ms")
                    if update['ttft_ms']:
                        status_parts.append(f"TTFT-Gen: {update['ttft_ms']:.1f}ms")
                    if update['ttft_text_ms']:
                        status_parts.append(f"TTFT-Text: {update['ttft_text_ms']:.1f}ms")
                    
                    status_parts.append(f"Audio segments: {segment_count}")
                    status_text = " | ".join(status_parts)
                    
                    # Update progress
                    progress(update['total_tokens'] / 100, desc=f"Generated {update['total_tokens']} tokens, {segment_count} audio segments")
                    
                    yield current_text, status_text, audio_output, latest_audio_segment, segment_count
                    
                elif update['type'] == 'audio_segment_complete':
                    current_text = update['text']
                    latest_audio_segment = update['audio_file']
                    segment_count = update['segment_count']
                    
                    # Build status for audio segment completion
                    status_parts = [f"🎵 Audio segment {segment_count} complete!"]
                    status_parts.append(f"Tokens: {update['total_tokens']}")
                    
                    if update.get('ttft_from_server_ms'):
                        status_parts.append(f"TTFT-Server: {update['ttft_from_server_ms']:.1f}ms")
                    if update['audio_encode_time_ms']:
                        status_parts.append(f"Audio-Encode: {update['audio_encode_time_ms']:.1f}ms")
                    if update['ttft_ms']:
                        status_parts.append(f"TTFT-Gen: {update['ttft_ms']:.1f}ms")
                    if update['ttfat_ms']:
                        status_parts.append(f"TTFAT: {update['ttfat_ms']:.1f}ms")
                    
                    status_text = " | ".join(status_parts)
                    
                    yield current_text, status_text, audio_output, latest_audio_segment, segment_count
                    
                elif update['type'] == 'final_result':
                    current_text = update['full_text']
                    audio_output = update['audio_files'][-1] if update['audio_files'] else None
                    segment_count = update['segment_count']
                    
                    # Build final status
                    status_parts = [f"✅ Complete! {update['total_tokens']} tokens in {update['generation_time_s']:.2f}s"]
                    status_parts.append(f"Audio segments: {segment_count}")
                    
                    if update.get('ttft_from_server_ms'):
                        status_parts.append(f"TTFT-Server: {update['ttft_from_server_ms']:.1f}ms")
                    if update['audio_encode_time_ms']:
                        status_parts.append(f"Audio-Encode: {update['audio_encode_time_ms']:.1f}ms")
                    if update['ttft_ms']:
                        status_parts.append(f"TTFT-Gen: {update['ttft_ms']:.1f}ms")
                    if update['ttfat_ms']:
                        status_parts.append(f"TTFAT: {update['ttfat_ms']:.1f}ms")
                    
                    status_text = " | ".join(status_parts)
                    
                    yield current_text, status_text, audio_output, latest_audio_segment, segment_count
                    
        except Exception as e:
            error_msg = f"❌ Error during real-time streaming: {str(e)}"
            log_event("UI_ERROR", "Spoken QA", error_msg)
            yield error_msg, f"Error: {str(e)}", None, None, 0
    
    def clear_inputs():
        """Clear all inputs and outputs"""
        return None, "", None, None, 0
    
    # Create Gradio interface
    with gr.Blocks(
        title="VITA-Audio Real-Time Streaming with Immediate Audio Decoding",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .center-title {
            text-align: center !important;
            margin-bottom: 2rem !important;
        }
        .info-panel {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        """
    ) as demo:
        
        # Centered hero title
        gr.HTML("""
        <div class="center-title">
            <h1>🎙️ VITA-Audio Real-Time Streaming with IMMEDIATE Audio Decoding</h1>
        </div>
        """)
        
        with gr.Row():
            # Left side - Main interface
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎤 Audio Input")
                        audio_input = gr.Audio(
                            label="Upload or Record Audio",
                            type="filepath",
                            interactive=True
                        )
                        
                        with gr.Row():
                            stream_btn = gr.Button("🎯 Start Real-Time Streaming", variant="primary", size="lg")
                            clear_btn = gr.Button("🧹 Clear", variant="secondary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Real-Time Text Stream")
                        text_output = gr.Textbox(
                            label="Streaming Text Response",
                            lines=6,
                            interactive=False,
                            show_copy_button=True
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎵 Latest Audio Segment (Immediate)")
                        streaming_audio_output = gr.Audio(
                            label="Latest Audio Segment (Generated in Real-Time)",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 🎵 Final Complete Audio")
                        final_audio_output = gr.Audio(
                            label="Final Complete Audio Response",
                            interactive=False
                        )
                
                gr.Markdown("### 📊 Real-Time Performance Metrics")
                status_output = gr.Textbox(
                    label="Real-Time Performance Metrics",
                    lines=4,
                    interactive=False
                )
                
                gr.Markdown("### 🔢 Audio Segment Counter")
                segment_counter_output = gr.Number(
                    label="Audio Segments Generated",
                    interactive=False
                )
            
            # Right side - Information panel
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="info-panel">
                    <h3>🚀 Real-Time Audio Decoding</h3>
                    <ul>
                        <li><strong>Immediate Processing:</strong> Audio tokens decoded as soon as segments complete</li>
                        <li><strong>No Waiting:</strong> Audio generation starts immediately, not after all tokens</li>
                        <li><strong>Streaming Audio:</strong> Listen to audio segments as they're generated</li>
                        <li><strong>Lower Latency:</strong> Faster audio response times</li>
                    </ul>
                </div>
                """)
                
                gr.HTML("""
                <div class="info-panel">
                    <h3>📊 Timing Metrics</h3>
                    <ul>
                        <li><strong>TTFT-Server:</strong> Time from audio hitting server to first token</li>
                        <li><strong>Audio-Encode:</strong> Time to encode input audio to tokens</li>
                        <li><strong>TTFT-Gen:</strong> Time from generation start to first token</li>
                        <li><strong>TTFAT:</strong> Time from generation start to first audio segment</li>
                    </ul>
                </div>
                """)
                
                gr.HTML("""
                <div class="info-panel">
                    <h3>📋 How It Works</h3>
                    <ol>
                        <li>Upload or record an audio file</li>
                        <li>Click "🎯 Start Real-Time Streaming"</li>
                        <li>Watch text appear word-by-word in real-time</li>
                        <li>Audio segments are generated IMMEDIATELY as they complete</li>
                        <li>Listen to audio segments as they're generated</li>
                        <li>Get final complete audio response</li>
                    </ol>
                </div>
                """)
        
        # Event handlers
        stream_btn.click(
            fn=process_audio_realtime_streaming,
            inputs=[audio_input],
            outputs=[text_output, status_output, final_audio_output, streaming_audio_output, segment_counter_output],
            show_progress=True
        )
        
        clear_btn.click(
            fn=clear_inputs,
            outputs=[audio_input, text_output, status_output, final_audio_output, streaming_audio_output]
        )
    
    return demo

def main():
    """Main function to initialize and launch the real-time streaming demo"""
    
    main_start_time = get_monotonic_time()
    log_event("STARTUP", "SYSTEM", "Starting VITA-Audio Real-Time Streaming Demo with Immediate Audio Decoding...",
              start_time=f"{main_start_time:.6f}")
    
    try:
        # Initialize streaming S2S inference engine
        engine_start_time = get_monotonic_time()
        log_event("ENGINE_INIT_START", "SYSTEM", "Starting S2S inference engine initialization")
        
        s2s_engine = StreamingS2SInference(
            model_name_or_path=model_name_or_path,
            audio_tokenizer_path=audio_tokenizer_path,
            audio_tokenizer_type=audio_tokenizer_type,
            flow_path=flow_path,
            audio_tokenizer_rank=0
        )
        
        engine_end_time = get_monotonic_time()
        log_latency_metric("ENGINE_INIT", engine_start_time, engine_end_time, "SYSTEM")
        log_event("STARTUP", "SYSTEM", "Real-time streaming S2S engine created successfully")
        
        # Create and launch interface
        interface_start_time = get_monotonic_time()
        log_event("INTERFACE_CREATE_START", "SYSTEM", "Creating Gradio interface")
        
        demo = create_realtime_streaming_interface(s2s_engine)
        
        interface_end_time = get_monotonic_time()
        log_latency_metric("INTERFACE_CREATE", interface_start_time, interface_end_time, "SYSTEM")
        
        log_event("STARTUP", "SYSTEM", "Launching real-time streaming demo with immediate audio decoding")
        
        # Launch the demo
        launch_start_time = get_monotonic_time()
        demo.launch(
            server_name="0.0.0.0",
            server_port=5009,
            share=True,
            show_error=True,
            quiet=False
        )
        
        launch_end_time = get_monotonic_time()
        log_latency_metric("DEMO_LAUNCH", launch_start_time, launch_end_time, "SYSTEM")
        
        # Log total startup time
        main_end_time = get_monotonic_time()
        log_latency_metric("TOTAL_STARTUP", main_start_time, main_end_time, "SYSTEM")
        
    except Exception as e:
        main_end_time = get_monotonic_time()
        log_latency_metric("STARTUP_FAILED", main_start_time, main_end_time, "SYSTEM",
                          error=str(e))
        log_event("STARTUP", "SYSTEM", f"Failed to start demo: {str(e)}")
        raise

if __name__ == "__main__":
    main()
