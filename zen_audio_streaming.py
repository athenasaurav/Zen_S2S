import math
import os
import sys
import time
import warnings
import re
import queue
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import asyncio

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

def get_utc_timestamp():
    """Get current UTC timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()

def log_event(event_type, task_type, message, **kwargs):
    """Log event with UTC timestamp and task context"""
    timestamp = get_utc_timestamp()
    task_info = f"[{task_type}]" if task_type else ""
    print(f"{timestamp} {task_info} {event_type}: {message}")
    
    # Log additional kwargs if provided
    for key, value in kwargs.items():
        if value is not None:
            print(f"{timestamp} {task_info} {event_type}: {key}: {value}")

def is_wav(file_path):
    if file_path is None:
        return False
    wav_extensions = [".wav", ".mp3", ".flac", ".ogg"]
    _, ext = os.path.splitext(file_path)
    return ext.lower() in wav_extensions

def extract_audio_tokens_from_text(text):
    """Extract individual audio tokens from text as they appear"""
    pattern = re.compile(r"<\|audio_(\d+)\|>")
    matches = pattern.finditer(text)
    tokens = []
    for match in matches:
        token_id = int(match.group(1))
        start_pos = match.start()
        end_pos = match.end()
        tokens.append({
            'token_id': token_id,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'raw_token': match.group(0)
        })
    return tokens

def clean_text_for_display(text):
    """Clean text by removing audio tokens but keeping readable text"""
    # Remove audio tokens but keep the text
    clean_text = re.sub(r"<\|audio_\d+\|>", "", text)
    clean_text = re.sub(r"<\|begin_of_audio\|>|<\|end_of_audio\|>", "", clean_text)
    clean_text = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", clean_text)
    clean_text = re.sub(r"(system|user|assistant)\s*", "", clean_text)
    return clean_text.strip()

class CriticalAudioProcessor:
    """CRITICAL FIX: Handles audio token decoding with correct offset calculation"""
    
    def __init__(self, audio_tokenizer, audio_offset):
        self.audio_tokenizer = audio_tokenizer
        self.audio_offset = audio_offset
        self.audio_segments = []
        self.audio_files = []
        self.processing_thread = None
        self.audio_queue = queue.Queue()
        self.is_running = True
        self.segment_counter = 0
        self.all_audio_tokens = []  # Store all tokens for final processing
        self.all_audio_token_strings = []  # Store token strings for display
        
        # CRITICAL: Determine correct offset calculation method
        log_event("AUDIO_PROCESSOR_INIT", "REALTIME", f"Initializing with audio_offset: {audio_offset}")
        
    def start_processing(self):
        """Start background audio processing thread"""
        def process_audio_tokens():
            while self.is_running:
                try:
                    token_data = self.audio_queue.get(timeout=0.5)
                    if token_data is None:  # Shutdown signal
                        break
                    
                    if token_data.get('final_process', False):
                        # Process all collected tokens at once for best quality
                        self.process_final_audio()
                    else:
                        # Just collect tokens for now
                        self.all_audio_tokens.append(token_data['token_id'])
                        self.all_audio_token_strings.append(token_data['token_string'])
                        log_event("AUDIO_TOKEN_COLLECTED", "REALTIME", 
                                f"Collected audio token {token_data['token_id']}, total: {len(self.all_audio_tokens)}")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    log_event("AUDIO_PROCESS_ERROR", "REALTIME", f"Error processing audio: {e}")
        
        self.processing_thread = threading.Thread(target=process_audio_tokens, daemon=True)
        self.processing_thread.start()
        log_event("AUDIO_PROCESSOR", "REALTIME", "Started CRITICAL audio processing thread")
    
    def add_audio_token(self, token_id, token_string):
        """Add audio token for processing"""
        if self.is_running:
            self.audio_queue.put({
                'token_id': token_id, 
                'token_string': token_string,
                'final_process': False
            })
    
    def process_final_audio(self):
        """CRITICAL FIX: Process all collected audio tokens with correct offset"""
        if not self.all_audio_tokens:
            log_event("FINAL_AUDIO_DECODE", "REALTIME", "No audio tokens to process")
            return None
            
        try:
            decode_start = time.time()
            
            log_event("FINAL_AUDIO_DECODE", "REALTIME", f"Decoding {len(self.all_audio_tokens)} audio tokens")
            log_event("FINAL_AUDIO_DECODE", "REALTIME", f"Sample tokens: {self.all_audio_tokens[:5]}")
            log_event("FINAL_AUDIO_DECODE", "REALTIME", f"Audio offset: {self.audio_offset}")
            
            # CRITICAL FIX: Try different offset calculation methods
            # Method 1: Direct token IDs (no offset subtraction)
            try:
                log_event("FINAL_AUDIO_DECODE", "REALTIME", "Trying Method 1: Direct token IDs")
                audio_segment = self.audio_tokenizer.decode(self.all_audio_tokens)
                if audio_segment is not None and len(audio_segment) > 0:
                    decode_time = (time.time() - decode_start) * 1000
                    return self._save_audio_segment(audio_segment, decode_time, "method1_direct")
            except Exception as e:
                log_event("FINAL_AUDIO_DECODE", "REALTIME", f"Method 1 failed: {e}")
            
            # Method 2: Subtract offset (original approach)
            try:
                log_event("FINAL_AUDIO_DECODE", "REALTIME", "Trying Method 2: Subtract offset")
                adjusted_tokens = [token_id - self.audio_offset for token_id in self.all_audio_tokens]
                log_event("FINAL_AUDIO_DECODE", "REALTIME", f"Adjusted tokens sample: {adjusted_tokens[:5]}")
                
                # Check if adjusted tokens are valid (positive)
                if all(token >= 0 for token in adjusted_tokens):
                    audio_segment = self.audio_tokenizer.decode(adjusted_tokens)
                    if audio_segment is not None and len(audio_segment) > 0:
                        decode_time = (time.time() - decode_start) * 1000
                        return self._save_audio_segment(audio_segment, decode_time, "method2_offset")
                else:
                    log_event("FINAL_AUDIO_DECODE", "REALTIME", "Method 2: Negative tokens after offset subtraction")
            except Exception as e:
                log_event("FINAL_AUDIO_DECODE", "REALTIME", f"Method 2 failed: {e}")
            
            log_event("FINAL_AUDIO_DECODE", "REALTIME", "All decoding methods failed")
                    
        except Exception as e:
            log_event("AUDIO_DECODE_ERROR", "REALTIME", f"Critical error in audio decoding: {e}")
        
        return None
    
    def _save_audio_segment(self, audio_segment, decode_time, method_name):
        """Save audio segment to file"""
        try:
            timestamp = int(time.time() * 1000)
            audio_file = f"/tmp/zen_final_{method_name}_{timestamp}.wav"
            
            import soundfile as sf
            sf.write(audio_file, audio_segment, 16000)
            
            self.audio_files.append(audio_file)
            self.audio_segments.append(audio_segment)
            
            log_event("FINAL_AUDIO_SAVED", "REALTIME", f"Final audio saved using {method_name}", 
                      tokens=len(self.all_audio_tokens),
                      decode_time_ms=f"{decode_time:.1f}",
                      file=os.path.basename(audio_file),
                      duration_s=f"{len(audio_segment)/16000:.2f}")
            
            return audio_file
            
        except Exception as e:
            log_event("AUDIO_SAVE_ERROR", "REALTIME", f"Error saving audio: {e}")
            return None
    
    def finalize_audio(self):
        """Signal to process all collected tokens"""
        if self.is_running and self.all_audio_tokens:
            self.audio_queue.put({'final_process': True})
            # Wait a bit for processing
            time.sleep(2.0)
    
    def get_final_audio_file(self):
        """Get the final complete audio file"""
        return self.audio_files[-1] if self.audio_files else None
    
    def get_token_count(self):
        """Get current token count"""
        return len(self.all_audio_tokens)
    
    def get_all_audio_tokens_display(self):
        """Get all audio tokens for display"""
        if not self.all_audio_token_strings:
            return "No audio tokens generated"
        
        # Show all tokens, not just latest 10
        if len(self.all_audio_token_strings) <= 20:
            return f"All {len(self.all_audio_token_strings)} tokens: {', '.join(self.all_audio_token_strings)}"
        else:
            first_10 = ', '.join(self.all_audio_token_strings[:10])
            last_10 = ', '.join(self.all_audio_token_strings[-10:])
            return f"First 10: {first_10}\\n...\\nLast 10: {last_10}\\nTotal: {len(self.all_audio_token_strings)} tokens"
    
    def shutdown(self):
        """Shutdown audio processor"""
        self.is_running = False
        self.audio_queue.put(None)  # Shutdown signal
        if self.processing_thread:
            self.processing_thread.join(timeout=3.0)

class StreamingS2SInference:
    """Streaming Speech-to-Speech Inference with CRITICAL FIXES"""
    
    def __init__(self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, flow_path, audio_tokenizer_rank=0):
        log_event("INIT", "SYSTEM", "Starting Zen-Speech-To-Speech Streaming Demo...")
        
        self.model_name_or_path = model_name_or_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.audio_tokenizer_type = audio_tokenizer_type
        self.flow_path = flow_path
        self.audio_tokenizer_rank = audio_tokenizer_rank
        
        # Load tokenizer
        log_event("INIT", "SYSTEM", "Loading tokenizer...")
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        load_time = time.time() - start_time
        log_event("INIT", "SYSTEM", "Tokenizer loaded", load_time_s=f"{load_time:.2f}")
        
        # Load model
        log_event("INIT", "SYSTEM", "Loading model...")
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        load_time = time.time() - start_time
        log_event("INIT", "SYSTEM", "Model loaded", load_time_s=f"{load_time:.2f}")
        
        # Load audio tokenizer
        log_event("INIT", "SYSTEM", "Loading GLM-4-Voice vocoder...")
        if AUDIO_MODULES_AVAILABLE:
            try:
                start_time = time.time()
                self.audio_tokenizer = get_audio_tokenizer(
                    audio_tokenizer_path,
                    audio_tokenizer_type,
                    flow_path=flow_path,
                    rank=audio_tokenizer_rank,
                )
                load_time = time.time() - start_time
                log_event("INIT", "SYSTEM", "GLM-4-Voice vocoder loaded", load_time_s=f"{load_time:.2f}")
            except Exception as e:
                log_event("INIT", "SYSTEM", "Error loading GLM-4-Voice vocoder", error=str(e))
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
        log_event("INIT", "SYSTEM", "Initialization completed", audio_offset=self.audio_offset)

    def stream_spoken_qa(self, audio_path, max_returned_tokens=2048):
        """Stream Spoken QA with CRITICAL FIXES for audio processing"""
        
        # TIMING: Start measuring from when audio hits the server
        server_start_time = time.time()
        log_event("STREAM_START", "Spoken QA", "Audio hit server - starting processing", 
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
                  template_time_ms=f"{template_time*1000:.1f}",
                  time_from_server_ms=f"{(time.time() - server_start_time)*1000:.1f}")

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
                      encode_time_ms=f"{encode_time*1000:.1f}",
                      time_from_server_ms=f"{(time.time() - server_start_time)*1000:.1f}")
            
            # Replace <|audio|> in the input with actual audio tokens
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            input_text = input_text.replace("<|audio|>", f"<|begin_of_audio|>{audio_tokens_str}<|end_of_audio|>")
            
            # Re-tokenize with audio tokens
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        else:
            encode_time = time.time() - audio_encode_start_time
            log_event("AUDIO_ENCODING_SKIPPED", "Spoken QA", "Audio encoding skipped", 
                      encode_time_ms=f"{encode_time*1000:.1f}",
                      time_from_server_ms=f"{(time.time() - server_start_time)*1000:.1f}")

        # Move to device
        input_ids = input_ids.to(self.model.device)

        # Initialize streaming components
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            timeout=30.0, 
            skip_prompt=True, 
            skip_special_tokens=False
        )
        
        # Initialize CRITICAL audio processor
        audio_processor = CriticalAudioProcessor(self.audio_tokenizer, self.audio_offset)
        audio_processor.start_processing()
        
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
        
        log_event("GENERATION_START", "Spoken QA", "Starting streaming generation",
                  time_from_server_ms=f"{(time.time() - server_start_time)*1000:.1f}")
        generation_start_time = time.time()
        
        # Start generation in background thread
        generation_thread = threading.Thread(
            target=self.model.generate, 
            kwargs=generation_kwargs
        )
        generation_thread.start()
        
        # Stream processing variables
        full_response = ""
        text_tokens = []
        all_audio_tokens = []
        first_token_time = None
        first_text_token_time = None
        first_audio_token_time = None
        first_audio_segment_time = None
        token_count = 0
        
        log_event("STREAM_PROCESSING", "Spoken QA", "Starting real-time token streaming")
        
        # Process tokens as they arrive
        for new_token in streamer:
            current_time = time.time()
            token_count += 1
            
            # Track first token (any type)
            if first_token_time is None:
                first_token_time = current_time
                ttft = (first_token_time - generation_start_time) * 1000
                ttft_from_server = (first_token_time - server_start_time) * 1000
                log_event("TTFT", "Spoken QA", f"First token generated", 
                          token=new_token.strip()[:50], 
                          ttft_ms=f"{ttft:.1f}",
                          ttft_from_server_ms=f"{ttft_from_server:.1f}")
            
            full_response += new_token
            
            # Check if this token contains audio tokens
            audio_tokens_in_token = extract_audio_tokens_from_text(new_token)
            
            if audio_tokens_in_token:
                # Track first audio token
                if first_audio_token_time is None:
                    first_audio_token_time = current_time
                    ttfat = (first_audio_token_time - generation_start_time) * 1000
                    ttfat_from_server = (first_audio_token_time - server_start_time) * 1000
                    log_event("TTFAT", "Spoken QA", f"First audio token generated", 
                              ttfat_ms=f"{ttfat:.1f}",
                              ttfat_from_server_ms=f"{ttfat_from_server:.1f}",
                              first_audio_token=audio_tokens_in_token[0]['raw_token'])
                
                # Process audio tokens
                for audio_token_info in audio_tokens_in_token:
                    token_id = audio_token_info['token_id']
                    token_string = audio_token_info['raw_token']
                    all_audio_tokens.append(token_id)
                    
                    # Track first audio segment creation
                    if first_audio_segment_time is None:
                        first_audio_segment_time = current_time
                        ttfas = (first_audio_segment_time - generation_start_time) * 1000
                        ttfas_from_server = (first_audio_segment_time - server_start_time) * 1000
                        log_event("TTFAS", "Spoken QA", f"First audio segment creation started", 
                                  ttfas_ms=f"{ttfas:.1f}",
                                  ttfas_from_server_ms=f"{ttfas_from_server:.1f}")
                    
                    # Add to CRITICAL audio processor
                    audio_processor.add_audio_token(token_id, token_string)
                    
                    log_event("AUDIO_TOKEN_STREAMING", "Spoken QA", f"Audio token streaming", 
                              token_id=token_id,
                              token=token_string,
                              total_audio_tokens=len(all_audio_tokens))
                
                # Yield streaming update with audio tokens
                yield {
                    'type': 'token_update',
                    'current_text': clean_text_for_display(full_response),
                    'current_audio_tokens': audio_processor.get_all_audio_tokens_display(),
                    'total_tokens': token_count,
                    'audio_tokens_count': len(all_audio_tokens),
                    'ttft_ms': (first_token_time - generation_start_time) * 1000 if first_token_time else None,
                    'ttft_text_ms': (first_text_token_time - generation_start_time) * 1000 if first_text_token_time else None,
                    'ttfat_ms': (first_audio_token_time - generation_start_time) * 1000 if first_audio_token_time else None,
                    'ttfas_ms': (first_audio_segment_time - generation_start_time) * 1000 if first_audio_segment_time else None,
                    'ttft_from_server_ms': (first_token_time - server_start_time) * 1000 if first_token_time else None,
                    'audio_encode_time_ms': encode_time * 1000
                }
            else:
                # This is a text token
                if first_text_token_time is None and new_token.strip() and not new_token.startswith('<|'):
                    first_text_token_time = current_time
                    ttft_text = (first_text_token_time - generation_start_time) * 1000
                    ttft_text_from_server = (first_text_token_time - server_start_time) * 1000
                    log_event("TTFT_TEXT", "Spoken QA", f"First text token generated", 
                              ttft_text_ms=f"{ttft_text:.1f}",
                              ttft_text_from_server_ms=f"{ttft_text_from_server:.1f}",
                              token=new_token.strip()[:50])
                
                text_tokens.append(new_token)
                
                log_event("TEXT_TOKEN_STREAMING", "Spoken QA", f"Text token streaming", 
                          token=new_token.strip()[:50],
                          text_token_count=len(text_tokens))
                
                # Yield streaming update with text
                yield {
                    'type': 'token_update',
                    'current_text': clean_text_for_display(full_response),
                    'current_audio_tokens': audio_processor.get_all_audio_tokens_display(),
                    'total_tokens': token_count,
                    'audio_tokens_count': len(all_audio_tokens),
                    'ttft_ms': (first_token_time - generation_start_time) * 1000 if first_token_time else None,
                    'ttft_text_ms': (first_text_token_time - generation_start_time) * 1000 if first_text_token_time else None,
                    'ttfat_ms': (first_audio_token_time - generation_start_time) * 1000 if first_audio_token_time else None,
                    'ttfas_ms': (first_audio_segment_time - generation_start_time) * 1000 if first_audio_segment_time else None,
                    'ttft_from_server_ms': (first_token_time - server_start_time) * 1000 if first_token_time else None,
                    'audio_encode_time_ms': encode_time * 1000
                }
        
        # Wait for generation to complete
        generation_thread.join()
        generation_end_time = time.time()
        
        # Process final audio with CRITICAL fixes
        log_event("FINAL_PROCESSING", "Spoken QA", "Processing final complete audio")
        audio_processor.finalize_audio()
        
        # Get final audio file
        final_audio_file = audio_processor.get_final_audio_file()
        
        # Shutdown audio processor
        audio_processor.shutdown()
        
        total_generation_time = generation_end_time - generation_start_time
        total_inference_time = generation_end_time - server_start_time
        
        log_event("GENERATION_COMPLETE", "Spoken QA", "Streaming generation completed", 
                  total_time_s=f"{total_generation_time:.2f}",
                  total_time_from_server_s=f"{total_inference_time:.2f}",
                  total_tokens=token_count,
                  audio_tokens=len(all_audio_tokens),
                  text_tokens=len(text_tokens))
        
        # Final result
        yield {
            'type': 'final_result',
            'full_text': clean_text_for_display(full_response),
            'full_audio_tokens': audio_processor.get_all_audio_tokens_display(),
            'audio_file': final_audio_file,
            'total_tokens': token_count,
            'generation_time_s': total_generation_time,
            'total_time_s': total_inference_time,
            'audio_encode_time_ms': encode_time * 1000,
            'ttft_ms': (first_token_time - generation_start_time) * 1000 if first_token_time else None,
            'ttft_text_ms': (first_text_token_time - generation_start_time) * 1000 if first_text_token_time else None,
            'ttfat_ms': (first_audio_token_time - generation_start_time) * 1000 if first_audio_token_time else None,
            'ttfas_ms': (first_audio_segment_time - generation_start_time) * 1000 if first_audio_segment_time else None,
            'ttft_from_server_ms': (first_token_time - server_start_time) * 1000 if first_token_time else None
        }

def create_streaming_interface(s2s_engine):
    """Create improved Gradio streaming interface with better layout"""
    
    def process_audio_streaming(audio_file, progress=gr.Progress()):
        """Process audio with improved streaming updates"""
        
        if audio_file is None:
            return "❌ Please upload or record an audio file", "", None, "No audio provided"
        
        log_event("UI_REQUEST", "Spoken QA", "Starting streaming request", 
                  audio_file=os.path.basename(audio_file))
        
        # Initialize display variables
        current_text = ""
        current_audio_tokens = ""
        status_text = "🎯 Processing audio input..."
        audio_output = None
        
        try:
            # Stream the inference
            for update in s2s_engine.stream_spoken_qa(audio_file):
                
                if update['type'] == 'token_update':
                    current_text = update['current_text']
                    current_audio_tokens = update['current_audio_tokens']
                    
                    # Build status with timing metrics
                    status_parts = [f"🔄 Streaming... {update['total_tokens']} tokens"]
                    
                    if update.get('ttft_from_server_ms'):
                        status_parts.append(f"TTFT-Server: {update['ttft_from_server_ms']:.1f}ms")
                    if update.get('audio_encode_time_ms'):
                        status_parts.append(f"Audio-Encode: {update['audio_encode_time_ms']:.1f}ms")
                    if update['ttft_ms']:
                        status_parts.append(f"TTFT-Gen: {update['ttft_ms']:.1f}ms")
                    if update['ttft_text_ms']:
                        status_parts.append(f"TTFT-Text: {update['ttft_text_ms']:.1f}ms")
                    if update['ttfat_ms']:
                        status_parts.append(f"TTFAT: {update['ttfat_ms']:.1f}ms")
                    if update['ttfas_ms']:
                        status_parts.append(f"TTFAS: {update['ttfas_ms']:.1f}ms")
                    
                    status_parts.append(f"Audio tokens: {update['audio_tokens_count']}")
                    status_text = " | ".join(status_parts)
                    
                    # Update progress
                    progress(update['total_tokens'] / 100, desc=f"Generated {update['total_tokens']} tokens")
                    
                    yield current_text, current_audio_tokens, audio_output, status_text
                    
                elif update['type'] == 'final_result':
                    current_text = update['full_text']
                    current_audio_tokens = update['full_audio_tokens']
                    audio_output = update['audio_file']
                    
                    # Build final status with timing metrics
                    status_parts = [f"✅ Complete! {update['total_tokens']} tokens in {update['generation_time_s']:.2f}s"]
                    
                    if update.get('ttft_from_server_ms'):
                        status_parts.append(f"TTFT-Server: {update['ttft_from_server_ms']:.1f}ms")
                    if update['audio_encode_time_ms']:
                        status_parts.append(f"Audio-Encode: {update['audio_encode_time_ms']:.1f}ms")
                    if update['ttft_ms']:
                        status_parts.append(f"TTFT-Gen: {update['ttft_ms']:.1f}ms")
                    if update['ttft_text_ms']:
                        status_parts.append(f"TTFT-Text: {update['ttft_text_ms']:.1f}ms")
                    if update['ttfat_ms']:
                        status_parts.append(f"TTFAT: {update['ttfat_ms']:.1f}ms")
                    if update['ttfas_ms']:
                        status_parts.append(f"TTFAS: {update['ttfas_ms']:.1f}ms")
                    
                    status_text = " | ".join(status_parts)
                    
                    yield current_text, current_audio_tokens, audio_output, status_text
                    
        except Exception as e:
            error_msg = f"❌ Error during streaming: {str(e)}"
            log_event("UI_ERROR", "Spoken QA", error_msg)
            yield error_msg, "", None, f"Error: {str(e)}"
    
    def clear_inputs():
        """Clear all inputs and outputs"""
        return None, "", "", None, "Ready for new input"
    
    # Create Gradio interface with improved layout
    with gr.Blocks(
        title="Zen-Speech-To-Speech Real-Time Streaming Demo",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .audio-player {
            width: 100% !important;
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
        .info-panel h3 {
            color: white !important;
            margin-top: 0 !important;
        }
        .info-panel ul {
            margin-bottom: 0 !important;
        }
        """
    ) as demo:
        
        # Centered hero title
        gr.HTML("""
        <div class="center-title">
            <h1>🎙️ Zen-Speech-To-Speech Real-Time Streaming Demo</h1>
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
                            stream_btn = gr.Button("🎯 Start Streaming", variant="primary", size="lg")
                            clear_btn = gr.Button("🧹 Clear", variant="secondary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Real-Time Text Stream")
                        text_output = gr.Textbox(
                            label="Streaming Text Response",
                            lines=6,
                            interactive=False,
                            show_copy_button=True
                        )
                
                gr.Markdown("### 🎵 Complete Audio Token Display")
                audio_tokens_output = gr.Textbox(
                    label="All Audio Tokens Generated",
                    lines=4,
                    interactive=False
                )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎵 Complete Audio Response")
                        final_audio_output = gr.Audio(
                            label="Final High-Quality Audio",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 📊 Performance Metrics")
                        status_output = gr.Textbox(
                            label="Real-Time Performance Metrics",
                            lines=4,
                            interactive=False
                        )
            
            # Right side - Information panel
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="info-panel">
                    <h3>📊 Timing Metrics Explained</h3>
                    <ul>
                        <li><strong>TTFT-Server:</strong> Time from audio hitting server to first token</li>
                        <li><strong>Audio-Encode:</strong> Time to encode input audio to tokens</li>
                        <li><strong>TTFT-Gen:</strong> Time from generation start to first token</li>
                        <li><strong>TTFT-Text:</strong> Time from generation start to first meaningful text</li>
                        <li><strong>TTFAT:</strong> Time from generation start to first audio token</li>
                        <li><strong>TTFAS:</strong> Time from generation start to first audio segment processing</li>
                    </ul>
                </div>
                """)
                
                gr.HTML("""
                <div class="info-panel">
                    <h3>📋 Instructions</h3>
                    <ol>
                        <li>Upload or record an audio file</li>
                        <li>Click "🎯 Start Streaming" to begin real-time processing</li>
                        <li>Watch text appear word-by-word in real-time</li>
                        <li>See ALL audio tokens being collected (not just summary)</li>
                        <li>Listen to the complete high-quality audio response</li>
                    </ol>
                </div>
                """)
        
        # Event handlers
        stream_btn.click(
            fn=process_audio_streaming,
            inputs=[audio_input],
            outputs=[text_output, audio_tokens_output, final_audio_output, status_output],
            show_progress=True
        )
        
        clear_btn.click(
            fn=clear_inputs,
            outputs=[audio_input, text_output, audio_tokens_output, final_audio_output, status_output]
        )
    
    return demo

def main():
    """Main function to initialize and launch the improved streaming demo"""
    
    log_event("STARTUP", "SYSTEM", "Starting Zen-Speech-To-Speech Real-Time Streaming Demo")
    
    try:
        # Initialize streaming S2S inference engine
        s2s_engine = StreamingS2SInference(
            model_name_or_path=model_name_or_path,
            audio_tokenizer_path=audio_tokenizer_path,
            audio_tokenizer_type=audio_tokenizer_type,
            flow_path=flow_path,
            audio_tokenizer_rank=0
        )
        
        log_event("STARTUP", "SYSTEM", "Zen-Speech-To-Speech streaming engine created successfully")
        
        # Check if soundfile is available for audio generation
        try:
            import soundfile as sf
            log_event("STARTUP", "SYSTEM", "soundfile available for audio generation")
        except ImportError:
            log_event("STARTUP", "SYSTEM", "soundfile not available - audio generation may fail")
        
        # Create and launch interface
        demo = create_streaming_interface(s2s_engine)
        
        log_event("STARTUP", "SYSTEM", "Launching improved real-time streaming demo interface")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=5006,
            share=True,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        log_event("STARTUP", "SYSTEM", f"Failed to start demo: {str(e)}")
        raise

if __name__ == "__main__":
    main()
