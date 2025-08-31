#!/usr/bin/env python3
"""
Multi-GPU VITA-Audio Implementation (Based on Working Code)
- GPU 0: Text generation (VITA-Audio model)
- GPU 1: Audio decoding (GLM4Voice tokenizer)

Expected Performance:
- Audio chunk time: 922ms → ~200ms (4-5x faster)
- Parallel text and audio processing
"""

import math
import os
import sys
import time
import warnings
import re
from datetime import datetime, timezone
from threading import Thread
from queue import Queue
import threading
import queue

import gradio as gr
import numpy as np
import torch
from numba import jit
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Add GLM-4-Voice paths (ENABLE GLM-4-Voice tokenizer)
if True:
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

def log_event(event_type, task_type, message, **kwargs):
    """Log event with UTC timestamp and task context"""
    timestamp = get_utc_timestamp()
    task_info = f"[{task_type}]" if task_type else ""
    print(f"{timestamp} {task_info} {event_type}: {message}")

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

def find_warmup_audio_files():
    """Find available audio files in assets folder for warmup"""
    asset_dir = "asset"
    warmup_files = []
    
    if os.path.exists(asset_dir):
        for file in os.listdir(asset_dir):
            if file.endswith(('.wav', '.mp3')):
                file_path = os.path.join(asset_dir, file)
                if os.path.getsize(file_path) > 1000:  # Skip very small files
                    warmup_files.append(file_path)
    
    return warmup_files[:3]  # Use first 3 files for warmup

def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0 and count % 2 == 1:
                line = line.replace("`", "\\`")
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")
                line = line.replace(" ", "&nbsp;")
                line = line.replace("*", "&#42;")
                line = line.replace("_", "&#95;")
                line = line.replace("-", "&#45;")
                line = line.replace(".", "&#46;")
                line = line.replace("!", "&#33;")
                line = line.replace("(", "&#40;")
                line = line.replace(")", "&#41;")
                line = line.replace("$", "&#36;")
            lines[i] = "<br>" + line
    return "".join(lines)

def find_audio_segments_regex(text):
    """Find all substrings between <|begin_of_audio|> and <|end_of_audio|> using regex."""
    pattern = re.compile(r"<\|begin_of_audio\|>(.*?)<\|end_of_audio\|>", re.DOTALL)
    segments = pattern.findall(text)
    return [segment.strip() for segment in segments]

def extract_token_ids_as_int(text):
    """Extract token IDs from audio segments and return as integers."""
    pattern = re.compile(r"<\|audio_(\d+)\|>")
    token_ids = pattern.findall(text)
    return [int(id) for id in token_ids if id.strip()]

def extract_assistant_audio_tokens_only(output_text, audio_offset):
    """Extract ONLY assistant's audio tokens, excluding user's audio tokens"""
    log_event("EXTRACT", "AUDIO_TOKENS", "Extracting ONLY assistant audio tokens...")
    
    # Find the assistant section
    assistant_marker = "<|im_start|>assistant"
    assistant_start = output_text.find(assistant_marker)
    
    if assistant_start == -1:
        log_event("EXTRACT", "AUDIO_TOKENS", "No assistant section found!")
        return []
    
    # Extract only the assistant's part
    assistant_section = output_text[assistant_start:]
    
    # Find all audio segments in assistant section only
    assistant_audio_segments = find_audio_segments_regex(assistant_section)
    
    # Extract token IDs from assistant's audio segments only
    assistant_audio_tokens = []
    for segment in assistant_audio_segments:
        tokens = extract_token_ids_as_int(segment)
        assistant_audio_tokens.extend(tokens)
    
    log_event("EXTRACT", "AUDIO_TOKENS", f"Total assistant audio tokens: {len(assistant_audio_tokens)}")
    return assistant_audio_tokens

def clean_text_display(text, task_type="Spoken QA"):
    """Enhanced text cleaning to remove system message artifacts and audio tokens"""
    
    log_event("CLEAN", task_type, f"Cleaning text for {task_type}")
    
    # Remove system/user/assistant markers
    clean_text = text
    clean_text = clean_text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    clean_text = re.sub(r"(system|user|assistant)\s*", "", clean_text)
    
    # Extract audio segments first (for counting)
    audio_segments = find_audio_segments_regex(clean_text)
    total_audio_tokens = sum(len(extract_token_ids_as_int(segment)) for segment in audio_segments)
    
    # Remove ALL audio-related tokens for clean display
    clean_text = re.sub(r"<\|begin_of_audio\|>.*?<\|end_of_audio\|>", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(r"<\|audio_\d+\|>", "", clean_text)
    clean_text = clean_text.replace("<|audio|>", "")
    
    # Remove system artifacts
    system_artifacts = [
        "You are a helpful AI assistant.",
        "You are a helpful AI assistant",
        "Convert the speech to text.",
        "Convert the text to speech.",
    ]
    
    for artifact in system_artifacts:
        clean_text = clean_text.replace(artifact, "")
    
    # Clean up whitespace
    clean_text = re.sub(r"\n\s*\n", "\n", clean_text)
    clean_text = re.sub(r"^\s+|\s+$", "", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)
    clean_text = re.sub(r"^[.\s,;:!?]+", "", clean_text)
    
    final_text = clean_text.strip()
    
    log_event("CLEAN", task_type, f"Final cleaned text: '{final_text}'")
    
    return final_text, len(audio_segments), total_audio_tokens

class MultiGPUAudioProcessor:
    """Audio processor that runs on dedicated GPU with background worker"""
    
    def __init__(self, audio_tokenizer, device="cuda:1", sample_rate=16000):
        self.audio_tokenizer = audio_tokenizer
        self.device = device
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        self.is_warmed_up = False
        
        log_event("MULTI_GPU", "AUDIO", f"Audio processor initialized on {device}")
    
    def start_worker(self):
        """Start background audio processing worker"""
        if self.is_running:
            return
            
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()
        log_event("MULTI_GPU", "AUDIO", "Audio worker thread started")
    
    def stop_worker(self):
        """Stop background audio processing worker"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        log_event("MULTI_GPU", "AUDIO", "Audio worker thread stopped")
    
    def _audio_worker(self):
        """Background worker for audio processing on dedicated GPU"""
        torch.cuda.set_device(self.device)
        
        while self.is_running:
            try:
                # Get audio processing job
                job = self.audio_queue.get(timeout=1.0)
                if job is None:  # Shutdown signal
                    break
                
                chunk_id, audio_tokens, start_time = job
                
                # Process audio on dedicated GPU
                with torch.cuda.device(self.device):
                    chunk_start_time = time.time()
                    
                    # Decode audio tokens
                    audio_chunk = self.audio_tokenizer.decode(audio_tokens)
                    
                    chunk_time = time.time() - chunk_start_time
                    total_time = time.time() - start_time
                
                # Return result
                result = {
                    'chunk_id': chunk_id,
                    'audio_chunk': audio_chunk,
                    'chunk_time': chunk_time,
                    'total_time': total_time,
                    'tokens': len(audio_tokens),
                    'audio_duration': audio_chunk.shape[0]/self.sample_rate if hasattr(audio_chunk, 'shape') else 0
                }
                
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                log_event("MULTI_GPU", "AUDIO", f"Audio worker error: {e}")
                continue
    
    def warmup_audio_decoder(self):
        """Pre-warm the audio decoder using assets folder files"""
        if self.is_warmed_up:
            log_event("MULTI_GPU", "WARMUP", "Audio decoder already warmed up")
            return
            
        log_event("MULTI_GPU", "WARMUP", "🔥 Starting multi-GPU audio decoder warmup...")
        warmup_start_time = time.time()
        
        # Set device for warmup
        torch.cuda.set_device(self.device)
        
        # Find warmup audio files
        warmup_files = find_warmup_audio_files()
        
        if not warmup_files:
            log_event("MULTI_GPU", "WARMUP", "⚠️ No warmup files found, creating dummy tokens")
            # Create dummy audio tokens for warmup
            dummy_tokens = [1000, 2000, 3000, 4000]  # 4 dummy tokens
            try:
                with torch.cuda.device(self.device):
                    warmup_chunk_start = time.time()
                    dummy_audio = self.audio_tokenizer.decode(dummy_tokens)
                    warmup_chunk_time = time.time() - warmup_chunk_start
                    log_event("MULTI_GPU", "WARMUP", f"Dummy warmup chunk: {len(dummy_tokens)} tokens → {dummy_audio.shape[0]/self.sample_rate:.2f}s audio in {warmup_chunk_time:.3f}s")
            except Exception as e:
                log_event("MULTI_GPU", "WARMUP", f"Dummy warmup failed: {e}")
        else:
            # Use real audio files for warmup
            for i, warmup_file in enumerate(warmup_files):
                try:
                    log_event("MULTI_GPU", "WARMUP", f"Warming up with: {os.path.basename(warmup_file)}")
                    
                    with torch.cuda.device(self.device):
                        # Encode audio to tokens
                        encode_start = time.time()
                        warmup_tokens = self.audio_tokenizer.encode(warmup_file)
                        encode_time = time.time() - encode_start
                        
                        # Take first 4 tokens for warmup chunk
                        warmup_chunk_tokens = warmup_tokens[:4] if len(warmup_tokens) >= 4 else warmup_tokens
                        
                        # Decode tokens to audio (this warms up the decoder)
                        decode_start = time.time()
                        warmup_audio = self.audio_tokenizer.decode(warmup_chunk_tokens)
                        decode_time = time.time() - decode_start
                        
                        log_event("MULTI_GPU", "WARMUP", f"Warmup {i+1}: encode={encode_time:.3f}s, decode={decode_time:.3f}s, tokens={len(warmup_chunk_tokens)}")
                        
                        # Only need one successful warmup
                        if decode_time < 0.8:  # If decode is fast, we're warmed up
                            break
                            
                except Exception as e:
                    log_event("MULTI_GPU", "WARMUP", f"Warmup file {warmup_file} failed: {e}")
                    continue
        
        warmup_total_time = time.time() - warmup_start_time
        self.is_warmed_up = True
        log_event("MULTI_GPU", "WARMUP", f"🔥 Multi-GPU audio decoder warmup completed in {warmup_total_time:.3f}s")
    
    def process_audio_chunk_async(self, chunk_id: int, audio_tokens: list, start_time: float):
        """Submit audio chunk for async processing"""
        job = (chunk_id, audio_tokens, start_time)
        self.audio_queue.put(job)
    
    def get_completed_chunk(self, timeout=0.1):
        """Get completed audio chunk if available"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class S2SInferenceMultiGPU:
    """Multi-GPU Speech-to-Speech Inference based on working code"""
    
    def __init__(self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, flow_path, 
                 audio_tokenizer_rank=0, use_turbo=True, enable_warmup=True,
                 text_gpu="cuda:0", audio_gpu="cuda:1"):
        
        self.model_name_or_path = model_name_or_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.audio_tokenizer_type = audio_tokenizer_type
        self.flow_path = flow_path
        self.audio_tokenizer_rank = audio_tokenizer_rank
        self.use_turbo = use_turbo
        self.enable_warmup = enable_warmup
        self.text_gpu = text_gpu
        self.audio_gpu = audio_gpu
        
        # Load text model on primary GPU
        self._load_text_model()
        
        # Load audio components on dedicated GPU
        self._load_audio_components()
        
        # Setup multi-GPU processing
        self._setup_multi_gpu_processing()
        
        log_event("MULTI_GPU", "INIT", f"Multi-GPU VITA-Audio initialized")
        log_event("MULTI_GPU", "INIT", f"Text model on {text_gpu}, Audio processing on {audio_gpu}")
    
    def _load_text_model(self):
        """Load text generation model on primary GPU"""
        log_event("MULTI_GPU", "INIT", f"Loading text model on {self.text_gpu}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        log_event("MULTI_GPU", "INIT", "Tokenizer loaded")
        
        # Load model on specific GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            device_map={
                "": self.text_gpu  # Force all layers to text GPU
            },
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        log_event("MULTI_GPU", "INIT", f"Text model loaded on {self.text_gpu}")
        
        # Configure generation mode
        self.configure_generation_mode()
        
        # Get audio offset
        self.audio_offset = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")
        log_event("MULTI_GPU", "INIT", f"Audio offset: {self.audio_offset}")
    
    def _load_audio_components(self):
        """Load audio components on dedicated GPU"""
        log_event("MULTI_GPU", "INIT", f"Loading audio components on {self.audio_gpu}...")
        
        # Set audio GPU
        torch.cuda.set_device(self.audio_gpu)
        
        # Load audio tokenizer (using exact same approach as working code)
        if AUDIO_MODULES_AVAILABLE:
            try:
                self.audio_tokenizer = get_audio_tokenizer(
                    self.audio_tokenizer_path,
                    self.audio_tokenizer_type,
                    flow_path=self.flow_path,
                    rank=self.audio_tokenizer_rank,
                )
                log_event("MULTI_GPU", "INIT", f"Audio tokenizer loaded on {self.audio_gpu}")
                
            except Exception as e:
                log_event("MULTI_GPU", "INIT", f"Error loading audio tokenizer: {e}")
                self.audio_tokenizer = None
        else:
            self.audio_tokenizer = None
    
    def _setup_multi_gpu_processing(self):
        """Setup multi-GPU processing pipeline"""
        if self.audio_tokenizer:
            # Initialize multi-GPU audio processor
            self.streaming_processor = MultiGPUAudioProcessor(
                self.audio_tokenizer, 
                device=self.audio_gpu
            )
            
            # Start background audio processing
            self.streaming_processor.start_worker()
            
            # 🔥 WARMUP OPTIMIZATION: Pre-warm audio decoder
            if self.enable_warmup:
                self.streaming_processor.warmup_audio_decoder()
            
            # Audio processing state
            self.pending_chunks = {}
            self.completed_chunks = {}
            self.chunk_counter = 0
            
            log_event("MULTI_GPU", "INIT", "Multi-GPU processing pipeline initialized")
        else:
            self.streaming_processor = None
    
    def configure_generation_mode(self):
        """Configure generation parameters based on mode"""
        
        # Base configuration
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
        
        if self.use_turbo:
            # TURBO MODE: [1, 10] for fastest generation (295 tokens/second)
            self.model.generation_config.mtp_inference_mode = [1, 10]
            log_event("MULTI_GPU", "INIT", "🚀 TURBO MODE enabled: [1, 10] - Expected ~295 tokens/second, ~3.4ms/token")
        else:
            # BOOST MODE: [1, 10, 4, 10] for balanced performance (170 tokens/second)
            self.model.generation_config.mtp_inference_mode = [1, 10, 4, 10]
            log_event("MULTI_GPU", "INIT", "⚡ BOOST MODE enabled: [1, 10, 4, 10] - Expected ~170 tokens/second, ~5.9ms/token")
            
        log_event("MULTI_GPU", "INIT", f"MTP inference mode: {self.model.generation_config.mtp_inference_mode}")
        
        if self.model.config.model_type == "hunyuan":
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
    
    def switch_mode(self, use_turbo):
        """Switch between Turbo and Boost modes dynamically"""
        self.use_turbo = use_turbo
        self.configure_generation_mode()
    
    def run_infer_streaming(self, audio_path=None, prompt_audio_path=None, message="", task_type="Spoken QA",
                           stream_stride=4, max_returned_tokens=4096, sample_rate=16000):
        """Multi-GPU streaming inference with real-time audio generation"""
        
        request_start_time = time.time()
        
        log_event("MULTI_GPU", task_type, f"🎬 Starting multi-GPU streaming inference")
        log_event("MULTI_GPU", task_type, f"Text GPU: {self.text_gpu}, Audio GPU: {self.audio_gpu}")
        log_event("MULTI_GPU", task_type, f"Mode: {'TURBO' if self.use_turbo else 'BOOST'}")
        log_event("MULTI_GPU", task_type, f"MTP config: {self.model.generation_config.mtp_inference_mode}")
        log_event("MULTI_GPU", task_type, f"Audio decoder warmed up: {self.streaming_processor.is_warmed_up if self.streaming_processor else False}")
        
        # Reset multi-GPU state
        if self.streaming_processor:
            self.pending_chunks = {}
            self.completed_chunks = {}
            self.chunk_counter = 0
        
        # Prepare messages (same as working code)
        if task_type == "TTS":
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + [
                {"role": "user", "content": f"Convert the text to speech.\n{message}"}
            ]
        elif task_type == "ASR":
            if audio_path is None:
                raise ValueError("ASR task requires audio_path")
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + [
                {"role": "user", "content": "Convert the speech to text.\n<|audio|>"}
            ]
        else:  # Spoken QA
            if audio_path:
                messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + [
                    {"role": "user", "content": "<|audio|>"}
                ]
            else:
                messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + [
                    {"role": "user", "content": message}
                ]

        # Apply chat template
        template_start_time = time.time()
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        template_time = time.time() - template_start_time
        log_event("MULTI_GPU", task_type, f"Chat template applied in {template_time:.3f}s")

        # Handle audio input processing (EXACT same as working code)
        audios = None
        audio_indices = None
        
        if (audio_path is not None or prompt_audio_path is not None) and self.audio_tokenizer:
            log_event("MULTI_GPU", task_type, f"Processing audio input with tokenizer...")
            
            # Check if audio tokenizer applies to user role
            if self.audio_tokenizer.apply_to_role("user", is_contiguous=True):
                log_event("MULTI_GPU", task_type, "Using contiguous codec for audio processing")
                # Contiguous codec
                audio_paths = []
                if audio_path is not None:
                    audio_paths.append(audio_path)
                if prompt_audio_path is not None:
                    audio_paths.append(prompt_audio_path)
                    
                try:
                    audio_process_start = time.time()
                    # EXACT same signature as working code
                    input_ids, audios, audio_indices = add_audio_input_contiguous(
                        input_ids, audio_paths, self.tokenizer, self.audio_tokenizer
                    )
                    audio_process_time = time.time() - audio_process_start
                    log_event("MULTI_GPU", task_type, f"Processed {len(audio_paths)} audio files in {audio_process_time:.3f}s")
                    
                except Exception as e:
                    log_event("MULTI_GPU", task_type, f"Audio processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    audios = None
                    audio_indices = None
                    
            elif self.audio_tokenizer.apply_to_role("user", is_discrete=True):
                log_event("MULTI_GPU", task_type, "Using discrete codec for audio processing")
                # Discrete codec (same as working code)
                if audio_path is not None:
                    try:
                        audio_encode_start = time.time()
                        audio_tokens = self.audio_tokenizer.encode(audio_path)
                        audio_encode_time = time.time() - audio_encode_start
                        audio_tokens_str = "".join([f"<|audio_{i}|>" for i in audio_tokens])
                        log_event("MULTI_GPU", task_type, f"Encoded audio to {len(audio_tokens)} tokens in {audio_encode_time:.3f}s")
                        
                        # Replace <|audio|> in the input with actual audio tokens
                        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                        input_text = input_text.replace("<|audio|>", f"<|begin_of_audio|>{audio_tokens_str}<|end_of_audio|>")
                        
                        # Re-tokenize with audio tokens
                        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
                        
                    except Exception as e:
                        log_event("MULTI_GPU", task_type, f"Discrete audio processing error: {e}")

        # Move to text GPU
        input_ids = input_ids.to(self.text_gpu)

        # Setup streaming generation
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=60.0,
            skip_prompt=True,
            skip_special_tokens=False
        )
        
        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_returned_tokens,
            "use_cache": True,
            "num_logits_to_keep": 1,
        }
        
        if audios is not None:
            generation_kwargs["audios"] = audios.to(self.text_gpu)
        if audio_indices is not None:
            generation_kwargs["audio_indices"] = audio_indices

        # Start generation in separate thread
        generation_start_time = time.time()
        generation_thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        generation_thread.start()
        
        # Process streaming tokens with multi-GPU audio processing
        generated_text = ""
        current_audio_tokens = []
        
        first_token_time = None
        first_text_token_time = None
        first_audio_token_time = None
        first_audio_chunk_time = None
        
        log_event("MULTI_GPU", task_type, "🎯 Starting real-time token processing...")
        
        for new_text in streamer:
            current_time = time.time()
            token_latency = current_time - generation_start_time
            
            # Track first token
            if first_token_time is None:
                first_token_time = token_latency
                log_event("MULTI_GPU", task_type, f"🎯 FIRST TOKEN at {first_token_time:.3f}s")
            
            # Track first text token
            if new_text.strip() and first_text_token_time is None:
                first_text_token_time = token_latency
                log_event("MULTI_GPU", task_type, f"📝 FIRST TEXT TOKEN at {first_text_token_time:.3f}s: '{new_text.strip()}'")
            
            # Extract audio tokens
            audio_tokens_in_text = re.findall(r'<\|audio_(\d+)\|>', new_text)
            
            for token_str in audio_tokens_in_text:
                if token_str.strip():
                    try:
                        audio_token_id = int(token_str)
                        current_audio_tokens.append(audio_token_id)
                        
                        # Track first audio token
                        if first_audio_token_time is None:
                            first_audio_token_time = token_latency
                            log_event("MULTI_GPU", task_type, f"🎵 FIRST AUDIO TOKEN at {first_audio_token_time:.3f}s")
                        
                    except ValueError:
                        continue
            
            # Process audio chunks when we have enough tokens (multi-GPU)
            if len(current_audio_tokens) >= stream_stride and self.streaming_processor:
                chunk_tokens = current_audio_tokens[:stream_stride]
                current_audio_tokens = current_audio_tokens[stream_stride:]
                
                # Submit for async audio processing on GPU 1
                self.streaming_processor.process_audio_chunk_async(
                    self.chunk_counter, 
                    chunk_tokens, 
                    request_start_time
                )
                
                self.pending_chunks[self.chunk_counter] = {
                    'tokens': chunk_tokens,
                    'submit_time': current_time
                }
                
                self.chunk_counter += 1
            
            # Check for completed audio chunks
            self._process_completed_chunks(task_type, request_start_time)
            
            generated_text += new_text
        
        # Process remaining audio tokens
        if current_audio_tokens and self.streaming_processor:
            self.streaming_processor.process_audio_chunk_async(
                self.chunk_counter, 
                current_audio_tokens, 
                request_start_time
            )
            
            self.pending_chunks[self.chunk_counter] = {
                'tokens': current_audio_tokens,
                'submit_time': time.time()
            }
            
            self.chunk_counter += 1
        
        # Wait for all audio chunks to complete
        self._wait_for_all_chunks(task_type)
        
        # Generation complete
        generation_thread.join(timeout=10.0)
        
        total_time = time.time() - request_start_time
        generation_time = time.time() - generation_start_time
        
        # Log performance summary
        log_event("MULTI_GPU", task_type, "=== MULTI-GPU TIMING RESULTS ===")
        log_event("MULTI_GPU", task_type, f"Text GPU: {self.text_gpu}")
        log_event("MULTI_GPU", task_type, f"Audio GPU: {self.audio_gpu}")
        log_event("MULTI_GPU", task_type, f"🎯 First token latency: {first_token_time:.3f}s")
        if first_text_token_time:
            log_event("MULTI_GPU", task_type, f"📝 First text token latency: {first_text_token_time:.3f}s")
        if first_audio_token_time:
            log_event("MULTI_GPU", task_type, f"🎵 First audio token latency: {first_audio_token_time:.3f}s")
        if first_audio_chunk_time:
            log_event("MULTI_GPU", task_type, f"🎵 First audio chunk latency: {first_audio_chunk_time:.3f}s ⚡ (multi-GPU optimized!)")
        log_event("MULTI_GPU", task_type, f"🎵 Audio chunks generated: {len(self.completed_chunks)}")
        log_event("MULTI_GPU", task_type, f"Total request time: {total_time:.3f}s")
        log_event("MULTI_GPU", task_type, f"Generation time: {generation_time:.3f}s")
        log_event("MULTI_GPU", task_type, "=== END MULTI-GPU TIMING ===")
        
        return clean_text_display(generated_text, task_type)[0]
    
    def _process_completed_chunks(self, task_type: str, request_start_time: float):
        """Process completed audio chunks from background worker"""
        while True:
            result = self.streaming_processor.get_completed_chunk()
            if result is None:
                break
            
            chunk_id = result['chunk_id']
            chunk_time = result['chunk_time']
            tokens = result['tokens']
            audio_duration = result['audio_duration']
            
            # Track first audio chunk time
            if chunk_id == 0:
                first_audio_chunk_time = time.time() - request_start_time
                log_event("MULTI_GPU", task_type, f"🎵 FIRST AUDIO CHUNK at {first_audio_chunk_time:.3f}s ⚡ (multi-GPU optimized!)")
            
            # Log chunk completion
            performance_note = " ⚡ (multi-GPU optimized!)"
            log_event("MULTI_GPU", "STREAM", 
                     f"Chunk {chunk_id}: {tokens} tokens → {audio_duration:.2f}s audio in {chunk_time:.3f}s{performance_note}")
            
            # Store completed chunk
            self.completed_chunks[chunk_id] = result
            
            # Remove from pending
            if chunk_id in self.pending_chunks:
                del self.pending_chunks[chunk_id]
    
    def _wait_for_all_chunks(self, task_type: str, timeout: float = 30.0):
        """Wait for all pending audio chunks to complete"""
        wait_start = time.time()
        
        while self.pending_chunks and (time.time() - wait_start) < timeout:
            self._process_completed_chunks(task_type, wait_start)
            time.sleep(0.01)  # Small delay
        
        if self.pending_chunks:
            log_event("MULTI_GPU", task_type, f"Warning: {len(self.pending_chunks)} chunks still pending after timeout")
    
    def __del__(self):
        """Cleanup multi-GPU resources"""
        if hasattr(self, 'streaming_processor') and self.streaming_processor:
            self.streaming_processor.stop_worker()

def create_gradio_interface():
    """Create Gradio interface for multi-GPU VITA-Audio"""
    
    # Initialize multi-GPU engine
    log_event("MULTI_GPU", "MAIN", "🔥 Initializing Multi-GPU VITA-Audio Engine...")
    
    s2s_engine = S2SInferenceMultiGPU(
        model_name_or_path=model_name_or_path,
        audio_tokenizer_path=audio_tokenizer_path,
        audio_tokenizer_type=audio_tokenizer_type,
        flow_path=flow_path,
        use_turbo=True,  # Start with Turbo mode
        enable_warmup=True,  # Enable warmup
        text_gpu="cuda:0",
        audio_gpu="cuda:1"
    )
    
    log_event("MULTI_GPU", "MAIN", "✅ Multi-GPU Engine initialized successfully!")
    
    def chat_interface(audio_input, task_selector, use_turbo_mode, history):
        """Main chat interface function"""
        if audio_input is None:
            return history, ""
        
        # Switch mode if needed
        s2s_engine.switch_mode(use_turbo_mode)
        
        mode_name = "TURBO" if use_turbo_mode else "BOOST"
        log_event("MULTI_GPU", task_selector, f"Using {mode_name} mode (🔥 Multi-GPU)")
        
        try:
            # Run multi-GPU inference
            response = s2s_engine.run_infer_streaming(
                audio_path=audio_input,
                task_type=task_selector
            )
            
            # Update chat history
            history.append([f"[Audio Input] {os.path.basename(audio_input)}", response])
            
            return history, ""
            
        except Exception as e:
            error_msg = f"Multi-GPU processing error: {e}"
            log_event("MULTI_GPU", task_selector, error_msg)
            history.append([f"[Audio Input] {os.path.basename(audio_input)}", error_msg])
            return history, ""
    
    # Create Gradio interface
    with gr.Blocks(title="Multi-GPU VITA-Audio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Multi-GPU VITA-Audio (4x H100 Optimized)")
        gr.Markdown("**Text Generation: GPU:0 | Audio Processing: GPU:1 | Expected 4-5x faster audio chunks**")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Audio input
                audio_input = gr.Audio(
                    label="Audio Input", 
                    type="filepath",
                    format="wav"
                )
                
                # Task selector
                task_selector = gr.Dropdown(
                    choices=["Spoken QA", "ASR", "TTS"],
                    value="Spoken QA",
                    label="Task Type"
                )
                
                # Mode selector
                use_turbo_mode = gr.Checkbox(
                    label="🚀 Turbo Mode ([1,10] vs [1,10,4,10])",
                    value=True
                )
                
                # Submit button
                submit_btn = gr.Button("Submit", variant="primary")
            
            with gr.Column(scale=3):
                # Chat history
                chatbot = gr.Chatbot(
                    label="Multi-GPU Chat History", 
                    height=500
                )
                
                # Clear button
                clear_btn = gr.Button("Clear History")
        
        # Performance info
        gr.Markdown("""
        ### 🔥 Multi-GPU Performance Optimizations:
        - **GPU 0 (cuda:0)**: Text generation (VITA-Audio model)
        - **GPU 1 (cuda:1)**: Audio decoding (GLM4Voice tokenizer)  
        - **Expected**: 4-5x faster audio chunk generation (~200ms vs 900ms)
        - **Warmup**: Pre-warmed audio decoder using assets folder
        - **Parallel processing**: Text and audio generation simultaneously
        """)
        
        # Event handlers
        submit_btn.click(
            chat_interface,
            inputs=[audio_input, task_selector, use_turbo_mode, chatbot],
            outputs=[chatbot, audio_input]
        )
        
        clear_btn.click(
            lambda: ([], None),
            outputs=[chatbot, audio_input]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
