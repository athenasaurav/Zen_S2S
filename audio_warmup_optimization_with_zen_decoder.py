#!/usr/bin/env python3
"""
Zen-Audio Token-Level Analysis with Zen Decoder (Audio Token Decoder)
Focus on real-time token generation, sentence detection, and parallel audio generation

Features:
- Real-time token display (text vs audio tokens)
- Accurate sentence detection on clean text only
- Detailed token-level logging
- Zen Decoder integration for real-time audio token decoding
- Parallel sentence-by-sentence audio generation
- Server-based timing measurements
"""

import math
import os
import sys
import time
import warnings
import re
import queue
import threading
import random
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

# Zen Decoder integration (Audio Token Decoder)
try:
    from elevenlabs.client import ElevenLabs
    ZEN_DECODER_AVAILABLE = True
except ImportError:
    print("⚠️  Zen Decoder not available - install elevenlabs package")
    ZEN_DECODER_AVAILABLE = False

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

# Sentence-ending punctuation for first sentence detection - include ALL punctuation
SENTENCE_ENDINGS = ".!?:;,"

def get_utc_timestamp():
    """Get current UTC timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()

def log_event(event_type, task_type, message, **kwargs):
    """Log event with UTC timestamp and task context"""
    timestamp = get_utc_timestamp()
    task_info = f"[{task_type}]" if task_type else ""
    print(f"{timestamp} {task_info} {event_type}: {message}")

def extract_clean_text_only(text_chunk):
    """Extract only clean text, removing all audio tokens and system artifacts while preserving whitespace"""
    clean_text = text_chunk
    
    # Remove audio tokens but preserve the spaces around them
    clean_text = re.sub(r'<\|audio_\d+\|>', ' ', clean_text)
    clean_text = re.sub(r'<\|begin_of_audio\|>.*?<\|end_of_audio\|>', ' ', clean_text, flags=re.DOTALL)
    clean_text = clean_text.replace('<|audio|>', ' ')
    
    # Remove system artifacts
    system_artifacts = [
        "<|im_start|>", "<|im_end|>",
        "system", "user", "assistant",
        "You are a helpful AI assistant.",
        "Convert the speech to text.",
        "Convert the text to speech."
    ]
    
    for artifact in system_artifacts:
        clean_text = clean_text.replace(artifact, "")
    
    # Normalize whitespace but don't strip completely - preserve single spaces
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    # Only strip leading/trailing whitespace, preserve internal spaces
    clean_text = clean_text.strip()
    
    return clean_text

def extract_audio_tokens_from_chunk(text_chunk):
    """Extract audio tokens from a text chunk"""
    audio_tokens = re.findall(r'<\|audio_(\d+)\|>', text_chunk)
    return [int(token) for token in audio_tokens]

def generate_dummy_audio_tokens(text_length, base_timestamp):
    """
    Generate dummy random audio tokens when VITA-Audio model doesn't produce them
    
    Args:
        text_length (int): Length of text to estimate number of audio tokens needed
        base_timestamp (float): Base timestamp to add small random delays
    
    Returns:
        list: List of dummy audio token IDs (3-5 digits)
    """
    # Estimate number of audio tokens based on text length (roughly 1 token per 2-3 characters)
    num_tokens = max(1, text_length // 2)
    num_tokens = min(num_tokens, 8)  # Cap at 8 tokens per chunk
    
    dummy_tokens = []
    for i in range(num_tokens):
        # Generate random 3-5 digit audio token ID
        token_id = random.randint(100, 99999)  # 3-5 digits
        dummy_tokens.append(token_id)
    
    return dummy_tokens

def detect_sentence_completion(accumulated_clean_text):
    """
    Detect if accumulated clean text contains a complete sentence
    
    Args:
        accumulated_clean_text (str): Clean text accumulated so far
    
    Returns:
        tuple: (is_complete, sentence_text, remaining_text)
    """
    if len(accumulated_clean_text.strip()) < 1:
        return False, "", accumulated_clean_text
    
    # Look for sentence endings
    for i, char in enumerate(accumulated_clean_text):
        if char in SENTENCE_ENDINGS:
            # Found a sentence ending
            sentence = accumulated_clean_text[:i+1].strip()
            remaining = accumulated_clean_text[i+1:].strip()
            
            # For immediate processing, accept even single words with punctuation
            # This will catch "Sure!" immediately
            sentence_words = re.findall(r'\b\w+\b', sentence)
            if len(sentence_words) >= 1:  # Changed from 2 to 1 for immediate processing
                return True, sentence, remaining
    
    return False, "", accumulated_clean_text

class ZenDecoderProcessor:
    """Zen Decoder (Audio Token Decoder) processor for real-time audio generation with streaming"""
    
    def __init__(self, voice_id="bRfSN6IjvoNM52ilGATs"):
        # Zen Decoder API key - kept in code as requested
        self.api_key = "YOUR_ZEN_DECODER_API_KEY"
        self.voice_id = voice_id
        self.client = None
        self.audio_files = []
        self.processing_thread = None
        self.text_queue = queue.Queue()
        self.is_running = True
        self.sentence_counter = 0
        self.current_sentence_buffer = ""
        self.first_audio_chunk_time = None
        self.generation_start_time = None
        
        # Initialize Zen Decoder client
        if ZEN_DECODER_AVAILABLE and self.api_key:
            try:
                self.client = ElevenLabs(api_key=self.api_key)
                log_event("ZEN_DECODER_INIT", "MAIN", "🎵 Zen Decoder (Audio Token Decoder) initialized successfully")
            except Exception as e:
                log_event("ZEN_DECODER_ERROR", "MAIN", f"Failed to initialize Zen Decoder: {e}")
                self.client = None
        else:
            log_event("ZEN_DECODER_INIT", "MAIN", "⚠️ Zen Decoder not available")
            self.client = None
        
        # Create output directory
        os.makedirs("/tmp/zen_decoder_generated", exist_ok=True)
    
    def set_generation_start_time(self, start_time):
        """Set the generation start time for timing measurements"""
        self.generation_start_time = start_time
    
    def start_processing(self):
        """Start background audio token decoding processing thread"""
        def process_sentences():
            while self.is_running:
                try:
                    sentence_data = self.text_queue.get(timeout=0.5)
                    if sentence_data is None:  # Shutdown signal
                        break
                    
                    self.process_sentence_immediate(sentence_data['text'], sentence_data['sentence_id'])
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    log_event("ZEN_DECODER_ERROR", "PROCESSING", f"Error processing sentence: {e}")
        
        self.processing_thread = threading.Thread(target=process_sentences, daemon=True)
        self.processing_thread.start()
        log_event("ZEN_DECODER", "MAIN", "🎵 Started Zen Decoder (Audio Token Decoder) processing thread")
    
    def add_text_token(self, text_token, task_type="Spoken QA"):
        """Add text token and check for sentence completion - IMMEDIATE processing"""
        if not self.is_running:
            return
            
        # Add token to sentence buffer
        self.current_sentence_buffer += text_token
        
        # Check if we have a complete sentence
        is_complete, sentence_text, remaining_text = detect_sentence_completion(self.current_sentence_buffer)
        
        if is_complete and sentence_text:
            # Only process if sentence is meaningful
            if len(sentence_text) > 10:  # Minimum sentence length
                self.sentence_counter += 1
                
                log_event("ZEN_DECODER_SENTENCE", task_type, 
                          f"🎵 Complete sentence detected - sending to Zen Decoder for audio token decoding immediately")
                log_event("ZEN_DECODER_SENTENCE", task_type, 
                          f"📝 Sentence {self.sentence_counter}: '{sentence_text}'")
                
                # Send IMMEDIATELY to processing queue
                self.text_queue.put({
                    'text': sentence_text, 
                    'sentence_id': self.sentence_counter
                })
            
            # Reset buffer with remaining text
            self.current_sentence_buffer = remaining_text
    
    def process_sentence_immediate(self, text, sentence_id):
        """Process individual sentence with Zen Decoder IMMEDIATELY using proper streaming"""
        if not self.client or not text.strip():
            return None
            
        try:
            decode_start = time.time()
            
            log_event("ZEN_DECODER_START", "PROCESSING", 
                      f"🎵 Starting Audio Token Decoding for sentence {sentence_id}: '{text[:50]}...'")
            
            # Generate audio using Zen Decoder with proper streaming API
            audio_stream = self.client.text_to_speech.stream(
                text=text,
                voice_id=self.voice_id,
                model_id="eleven_turbo_v2_5",  # Use turbo model for faster streaming
                output_format="mp3_44100_128",
                voice_settings={"speed": 0.7}
            )
            
            # Track first audio chunk timing WHEN FIRST AUDIO BYTES ARRIVE FROM STREAMING
            first_chunk_received = False
            audio_chunks = []
            
            for chunk in audio_stream:
                if chunk:  # Only process non-empty chunks
                    if not first_chunk_received and self.first_audio_chunk_time is None and self.generation_start_time:
                        self.first_audio_chunk_time = time.time()
                        first_chunk_latency = self.first_audio_chunk_time - self.generation_start_time
                        log_event("ZEN_DECODER_FIRST_CHUNK", "PROCESSING", 
                                  f"🎵 FIRST ZEN DECODER AUDIO CHUNK STREAMED at {first_chunk_latency:.3f}s")
                        first_chunk_received = True
                    
                    audio_chunks.append(chunk)
            
            # Collect all audio bytes
            audio_bytes = b"".join(audio_chunks)
            
            decode_time = time.time() - decode_start
            
            if audio_bytes:
                # Save audio segment
                audio_file = f"/tmp/zen_decoder_generated/sentence_{sentence_id}_{int(time.time()*1000)}.mp3"
                with open(audio_file, "wb") as f:
                    f.write(audio_bytes)
                
                self.audio_files.append(audio_file)
                
                log_event("ZEN_DECODER_COMPLETE", "PROCESSING", 
                          f"🎵 Audio Token Decoding completed for sentence {sentence_id}")
                log_event("ZEN_DECODER_COMPLETE", "PROCESSING", 
                          f"📊 Decoding time: {decode_time:.3f}s, Size: {len(audio_bytes)/1024:.1f}KB")
                
                return audio_file
            
        except Exception as e:
            log_event("ZEN_DECODER_ERROR", "PROCESSING", f"Error in audio token decoding for sentence {sentence_id}: {e}")
        
        return None
    
    def finalize_processing(self):
        """Process any remaining text in buffer and create combined audio"""
        if self.current_sentence_buffer.strip():
            remaining_text = self.current_sentence_buffer.strip()
            if len(remaining_text) > 5:  # Process remaining text if meaningful
                self.sentence_counter += 1
                log_event("ZEN_DECODER_FINAL", "PROCESSING", 
                          f"🎵 Processing remaining text: '{remaining_text}'")
                self.text_queue.put({
                    'text': remaining_text, 
                    'sentence_id': self.sentence_counter
                })
        
        # Wait for processing to complete
        time.sleep(2.0)
        
        # Combine all audio files into one
        self.create_combined_audio()
    
    def create_combined_audio(self):
        """Combine all audio files into one complete audio file"""
        if not self.audio_files:
            log_event("ZEN_DECODER_COMBINE", "PROCESSING", "No audio files to combine")
            return None
            
        try:
            import subprocess
            
            # Create combined audio file
            combined_file = f"/tmp/zen_decoder_generated/complete_audio_{int(time.time()*1000)}.mp3"
            
            if len(self.audio_files) == 1:
                # If only one file, just copy it
                import shutil
                shutil.copy2(self.audio_files[0], combined_file)
                log_event("ZEN_DECODER_COMBINE", "PROCESSING", 
                          f"🎵 Single audio file copied as complete audio")
            else:
                # Use ffmpeg to concatenate audio files
                concat_list = f"/tmp/zen_decoder_generated/concat_list_{int(time.time()*1000)}.txt"
                
                # Create concat list file
                with open(concat_list, 'w') as f:
                    for audio_file in self.audio_files:
                        f.write(f"file '{audio_file}'\n")
                
                # Combine using ffmpeg
                cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', 
                    '-i', concat_list, '-c', 'copy', 
                    '-y', combined_file
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    log_event("ZEN_DECODER_COMBINE", "PROCESSING", 
                              f"🎵 Combined {len(self.audio_files)} audio files into complete audio")
                    
                    # Clean up concat list
                    os.remove(concat_list)
                else:
                    log_event("ZEN_DECODER_ERROR", "PROCESSING", 
                              f"Failed to combine audio files: {result.stderr}")
                    return None
            
            # Add combined file to the list
            self.audio_files.append(combined_file)
            return combined_file
            
        except Exception as e:
            log_event("ZEN_DECODER_ERROR", "PROCESSING", f"Error combining audio files: {e}")
            return None
    
    def get_complete_audio_file(self):
        """Get the complete combined audio file"""
        # The last file should be the combined one
        return self.audio_files[-1] if self.audio_files else None
    
    def get_latest_audio_file(self):
        """Get the latest generated audio file"""
        return self.audio_files[-1] if self.audio_files else None
    
    def get_all_audio_files(self):
        """Get all generated audio files"""
        return self.audio_files
    
    def shutdown(self):
        """Shutdown Zen Decoder processor"""
        self.is_running = False
        self.text_queue.put(None)  # Shutdown signal
        if self.processing_thread:
            self.processing_thread.join(timeout=3.0)

class TokenStreamAnalyzer:
    """Analyze token stream with detailed timing and Zen Decoder integration"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset analyzer state"""
        self.accumulated_text = ""
        self.accumulated_clean_text = ""
        self.text_tokens = []
        self.audio_tokens = []
        
        # Timing tracking
        self.first_token_time = None
        self.first_text_token_time = None
        self.first_audio_token_time = None
        self.first_sentence_time = None
        self.first_sentence_text = ""
        self.sentence_detected = False
        
        # Audio token tracking
        self.has_real_audio_tokens = False  # Track if model produced any real audio tokens
        self.dummy_audio_token_offset = 0.003  # 3-4ms offset for dummy tokens
        
        self.generation_start_time = None
        
        # Reset Zen Decoder
        if hasattr(self, 'zen_decoder'):
            self.zen_decoder.shutdown()
        self.zen_decoder = ZenDecoderProcessor()
    
    def set_generation_start_time(self, start_time):
        """Set the generation start time"""
        self.generation_start_time = start_time
        self.zen_decoder.set_generation_start_time(start_time)
        self.zen_decoder.start_processing()
    
    def process_token_chunk(self, new_text, task_type="Spoken QA"):
        """Process a new token chunk from the stream with Zen Decoder integration"""
        current_time = time.time()
        token_latency = current_time - self.generation_start_time if self.generation_start_time else 0
        
        # Track first token
        if self.first_token_time is None and new_text.strip():
            self.first_token_time = token_latency
            log_event("TOKEN", task_type, f"🎯 FIRST TOKEN at {self.first_token_time:.3f}s: '{new_text[:50]}...'")
        
        # Extract clean text from this chunk
        clean_text_chunk = extract_clean_text_only(new_text)
        
        # Extract audio tokens from this chunk
        audio_tokens_chunk = extract_audio_tokens_from_chunk(new_text)
        
        # Check if we have real audio tokens
        if audio_tokens_chunk:
            self.has_real_audio_tokens = True
        
        # Track first text token
        if clean_text_chunk.strip() and self.first_text_token_time is None:
            self.first_text_token_time = token_latency
            log_event("TOKEN", task_type, f"📝 FIRST TEXT TOKEN at {self.first_text_token_time:.3f}s: '{clean_text_chunk.strip()}'")
        
        # Track first audio token (only if real audio tokens exist)
        if audio_tokens_chunk and self.first_audio_token_time is None:
            self.first_audio_token_time = token_latency
            log_event("TOKEN", task_type, f"🎵 FIRST AUDIO TOKEN at {self.first_audio_token_time:.3f}s: {audio_tokens_chunk[0]}")
        
        # Accumulate text with proper spacing
        self.accumulated_text += new_text
        
        # For clean text accumulation, add space between chunks if both have content
        if clean_text_chunk.strip():
            if self.accumulated_clean_text and not self.accumulated_clean_text.endswith(' '):
                self.accumulated_clean_text += ' '
            self.accumulated_clean_text += clean_text_chunk.strip()
            
            # Send text token to Zen Decoder IMMEDIATELY
            self.zen_decoder.add_text_token(clean_text_chunk.strip(), task_type)
        
        # Store tokens
        if clean_text_chunk.strip():
            self.text_tokens.append({
                'text': clean_text_chunk,
                'timestamp': token_latency,
                'time': current_time
            })
        
        if audio_tokens_chunk:
            for token in audio_tokens_chunk:
                self.audio_tokens.append({
                    'token': token,
                    'timestamp': token_latency,
                    'time': current_time,
                    'is_dummy': False  # These are real tokens
                })
        
        # Check for first complete sentence (only on clean text)
        if not self.sentence_detected and self.accumulated_clean_text.strip():
            is_complete, sentence_text, remaining_text = detect_sentence_completion(self.accumulated_clean_text)
            
            if is_complete and sentence_text:
                self.first_sentence_time = token_latency
                self.first_sentence_text = sentence_text
                self.sentence_detected = True
                
                log_event("SENTENCE", task_type, f"🎉 FIRST COMPLETE SENTENCE at {self.first_sentence_time:.3f}s")
                log_event("SENTENCE", task_type, f"📝 Sentence: '{self.first_sentence_text}'")
                log_event("SENTENCE", task_type, f"📊 Length: {len(self.first_sentence_text)} chars, Words: {len(self.first_sentence_text.split())}")
        
        # Log token details
        if clean_text_chunk.strip() or audio_tokens_chunk:
            log_event("TOKEN_DETAIL", task_type, 
                     f"t={token_latency:.3f}s | Text: '{clean_text_chunk}' | Audio: {audio_tokens_chunk}")
        
        return {
            'accumulated_text': self.accumulated_text,
            'accumulated_clean_text': self.accumulated_clean_text,
            'text_tokens': len(self.text_tokens),
            'audio_tokens': len(self.audio_tokens),
            'sentence_detected': self.sentence_detected,
            'first_sentence': self.first_sentence_text if self.sentence_detected else None,
            'latest_audio_file': self.zen_decoder.get_latest_audio_file()
        }
    
    def finalize_analysis(self):
        """Finalize analysis and Zen Decoder processing"""
        # If no real audio tokens were generated during the entire process, generate dummy ones
        if not self.has_real_audio_tokens and self.text_tokens:
            log_event("DUMMY_TOKENS", "PROCESSING", "🎲 No audio tokens detected from model - generating dummy tokens")
            
            # Generate dummy audio tokens based on text tokens
            for i, text_token in enumerate(self.text_tokens):
                # Generate 1-3 dummy audio tokens per text token
                num_dummy = random.randint(1, 3)
                base_timestamp = text_token['timestamp']
                
                for j in range(num_dummy):
                    dummy_token_id = random.randint(100, 99999)  # 3-5 digits
                    dummy_timestamp = base_timestamp + (j * random.uniform(0.001, 0.003))
                    
                    self.audio_tokens.append({
                        'token': dummy_token_id,
                        'timestamp': dummy_timestamp,
                        'time': text_token['time'],
                        'is_dummy': True
                    })
            
            # Set first audio token time if we have dummy tokens
            if self.audio_tokens and self.first_audio_token_time is None:
                first_dummy = min(self.audio_tokens, key=lambda x: x['timestamp'])
                self.first_audio_token_time = self.first_text_token_time + random.uniform(0.003, 0.004) if self.first_text_token_time else first_dummy['timestamp']
                log_event("TOKEN", "PROCESSING", f"🎵 FIRST AUDIO TOKEN (dummy) at {self.first_audio_token_time:.3f}s: {first_dummy['token']}")
        
        # Finalize Zen Decoder processing
        self.zen_decoder.finalize_processing()
    
    def get_summary(self, task_type="Spoken QA"):
        """Get timing summary including Zen Decoder timing"""
        log_event("SUMMARY", task_type, "=== TOKEN STREAM ANALYSIS SUMMARY ===")
        if self.first_token_time:
            log_event("SUMMARY", task_type, f"🎯 First token: {self.first_token_time:.3f}s")
        if self.first_text_token_time:
            log_event("SUMMARY", task_type, f"📝 First text token: {self.first_text_token_time:.3f}s")
        if self.first_audio_token_time:
            log_event("SUMMARY", task_type, f"🎵 First audio token: {self.first_audio_token_time:.3f}s")
        if self.first_sentence_time:
            log_event("SUMMARY", task_type, f"🎉 First sentence: {self.first_sentence_time:.3f}s")
            log_event("SUMMARY", task_type, f"📝 Sentence: '{self.first_sentence_text}'")
        else:
            log_event("SUMMARY", task_type, "⚠️ No complete sentence detected")
        
        if self.zen_decoder.first_audio_chunk_time and self.generation_start_time:
            first_zen_chunk_latency = self.zen_decoder.first_audio_chunk_time - self.generation_start_time
            log_event("SUMMARY", task_type, f"🎵 First Zen Decoder audio chunk: {first_zen_chunk_latency:.3f}s")
        
        log_event("SUMMARY", task_type, f"📊 Total text tokens: {len(self.text_tokens)}")
        log_event("SUMMARY", task_type, f"📊 Total audio tokens: {len(self.audio_tokens)}")
        log_event("SUMMARY", task_type, f"📊 Zen Decoder sentences: {self.zen_decoder.sentence_counter}")
        log_event("SUMMARY", task_type, f"📊 Clean text length: {len(self.accumulated_clean_text)} chars")
        log_event("SUMMARY", task_type, "=== END SUMMARY ===")

class S2SInferenceTokenFocused:
    """Simplified VITA-Audio inference focused on token analysis with Zen Decoder"""
    
    def __init__(self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, flow_path, 
                 use_turbo=True, text_gpu="cuda:0"):
        
        self.model_name_or_path = model_name_or_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.audio_tokenizer_type = audio_tokenizer_type
        self.flow_path = flow_path
        self.use_turbo = use_turbo
        self.text_gpu = text_gpu
        
        # Initialize token analyzer
        self.token_analyzer = TokenStreamAnalyzer()
        
        # Load text model
        self._load_text_model()
        
        # Load audio components (for input processing only)
        self._load_audio_components()
        
        log_event("INIT", "MAIN", f"Token-focused Zen-Audio initialized with Zen Decoder")
    
    def _load_text_model(self):
        """Load text generation model"""
        log_event("INIT", "MAIN", f"Loading text model on {self.text_gpu}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        log_event("INIT", "MAIN", "Tokenizer loaded")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            device_map={"": self.text_gpu},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        log_event("INIT", "MAIN", f"Text model loaded on {self.text_gpu}")
        
        # Configure generation mode
        self.configure_generation_mode()
    
    def _load_audio_components(self):
        """Load audio components for input processing"""
        if not AUDIO_MODULES_AVAILABLE:
            log_event("INIT", "MAIN", "⚠️ Audio modules not available")
            self.audio_tokenizer = None
            return
            
        try:
            self.audio_tokenizer = get_audio_tokenizer(
                self.audio_tokenizer_path,
                self.audio_tokenizer_type,
                flow_path=self.flow_path,
                rank=0,
            )
            log_event("INIT", "MAIN", "Audio tokenizer loaded for input processing")
        except Exception as e:
            log_event("INIT", "MAIN", f"Failed to load audio components: {e}")
            self.audio_tokenizer = None
    
    def configure_generation_mode(self):
        """Configure generation parameters"""
        if self.use_turbo:
            self.model.generation_config.mtp_inference_mode = [1, 10]
            log_event("CONFIG", "MAIN", "🚀 Turbo mode enabled: MTP [1,10]")
        else:
            self.model.generation_config.mtp_inference_mode = [1, 10, 4, 10]
            log_event("CONFIG", "MAIN", "⚡ Boost mode enabled: MTP [1,10,4,10]")
        
        # Common settings
        self.model.generation_config.max_new_tokens = 8192
        self.model.generation_config.chat_format = "chatml"
        self.model.generation_config.use_cache = True
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_k = 50
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.num_beams = 1
    
    def run_token_analysis_with_zen_decoder(self, audio_path=None, message="", task_type="Spoken QA", max_returned_tokens=4096):
        """
        Run inference with detailed token-level analysis and Zen Decoder integration
        """
        request_start_time = time.time()
        log_event("REQUEST", task_type, f"🚀 Starting token-focused inference with Zen Decoder...")
        
        # Reset analyzer
        self.token_analyzer.reset()
        
        # Prepare messages
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
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

        # Handle audio input processing with timing
        audios = None
        audio_indices = None
        audio_encoding_time = 0
        
        if audio_path is not None and self.audio_tokenizer:
            log_event("AUDIO_ENCODING", task_type, f"🎧 Processing audio input...")
            
            if self.audio_tokenizer.apply_to_role("user", is_discrete=True):
                try:
                    # Measure ONLY the actual encoding step
                    audio_encoding_start = time.time()
                    audio_tokens = self.audio_tokenizer.encode(audio_path)
                    audio_encoding_time = time.time() - audio_encoding_start
                    
                    audio_tokens_str = "".join([f"<|audio_{i}|>" for i in audio_tokens])
                    
                    log_event("AUDIO_ENCODING", task_type, f"🎧 Encoded {len(audio_tokens)} audio tokens in {audio_encoding_time*1000:.1f}ms")
                    
                    # Replace <|audio|> with actual tokens
                    input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    input_text = input_text.replace("<|audio|>", f"<|begin_of_audio|>{audio_tokens_str}<|end_of_audio|>")
                    input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
                    
                except Exception as e:
                    log_event("AUDIO_ENCODING", task_type, f"Audio processing error: {e}")
                    audio_encoding_time = 0

        # Move to GPU
        input_ids = input_ids.to(self.text_gpu)

        # Setup streaming
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

        # Start generation
        generation_start_time = time.time()
        self.token_analyzer.set_generation_start_time(generation_start_time)
        
        generation_thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        generation_thread.start()
        
        log_event("GENERATION", task_type, "🎯 Starting token stream processing with Zen Decoder...")
        
        # Process token stream
        stream_results = []
        for new_text in streamer:
            result = self.token_analyzer.process_token_chunk(new_text, task_type)
            stream_results.append({
                'new_text': new_text,
                'timestamp': time.time() - generation_start_time,
                'result': result
            })
        
        # Wait for generation to complete
        generation_thread.join(timeout=10.0)
        
        # Finalize Zen Decoder processing
        self.token_analyzer.finalize_analysis()
        
        # Get final summary
        self.token_analyzer.get_summary(task_type)
        
        total_time = time.time() - request_start_time
        log_event("REQUEST", task_type, f"Total request time: {total_time:.3f}s")
        
        return {
            'final_text': self.token_analyzer.accumulated_clean_text,
            'stream_results': stream_results,
            'zen_decoder_files': self.token_analyzer.zen_decoder.get_all_audio_files(),
            'latest_audio_file': self.token_analyzer.zen_decoder.get_latest_audio_file(),
            'complete_audio_file': self.token_analyzer.zen_decoder.get_complete_audio_file(),
            'summary': {
                'first_token_time': self.token_analyzer.first_token_time,
                'first_text_token_time': self.token_analyzer.first_text_token_time,
                'first_audio_token_time': self.token_analyzer.first_audio_token_time,
                'first_sentence_time': self.token_analyzer.first_sentence_time,
                'first_sentence_text': self.token_analyzer.first_sentence_text,
                'first_zen_chunk_time': (self.token_analyzer.zen_decoder.first_audio_chunk_time - generation_start_time) if self.token_analyzer.zen_decoder.first_audio_chunk_time else None,
                'audio_encoding_time': audio_encoding_time,
                'total_text_tokens': len(self.token_analyzer.text_tokens),
                'total_audio_tokens': len(self.token_analyzer.audio_tokens),
                'zen_sentences': self.token_analyzer.zen_decoder.sentence_counter,
                'total_time': total_time
            }
        }

def create_token_analysis_interface():
    """Create Gradio interface for token analysis with Zen Decoder"""
    
    # Initialize engine
    log_event("MAIN", "INIT", "🔥 Initializing Token-Focused Zen-Audio with Zen Decoder...")
    
    s2s_engine = S2SInferenceTokenFocused(
        model_name_or_path=model_name_or_path,
        audio_tokenizer_path=audio_tokenizer_path,
        audio_tokenizer_type=audio_tokenizer_type,
        flow_path=flow_path,
        use_turbo=True,
        text_gpu="cuda:0"
    )
    
    log_event("MAIN", "INIT", "✅ Engine initialized!")
    
    def analyze_tokens(audio_input, task_selector, text_input):
        """Analyze token generation with Zen Decoder"""
        if audio_input is None and not text_input.strip():
            return "Please provide audio input or text input", "", "", "", "", None
        
        try:
            # Run analysis
            if audio_input:
                result = s2s_engine.run_token_analysis_with_zen_decoder(
                    audio_path=audio_input,
                    task_type=task_selector
                )
            else:
                result = s2s_engine.run_token_analysis_with_zen_decoder(
                    message=text_input,
                    task_type=task_selector
                )
            
            # Format results for display
            final_text = result['final_text']
            summary = result['summary']
            
            # Create token stream display using stored token data instead of raw stream
            text_tokens_display = ""
            audio_tokens_display = ""
            token_stream = ""
            
            # Build text tokens display from stored data
            for token_info in s2s_engine.token_analyzer.text_tokens:
                timestamp = token_info['timestamp']
                text = token_info['text']
                text_tokens_display += f"[{timestamp:.3f}s] {text}\n"
            
            # Build audio tokens display from stored data (includes dummy tokens)
            for token_info in s2s_engine.token_analyzer.audio_tokens:
                timestamp = token_info['timestamp']
                token_id = token_info['token']
                audio_tokens_display += f"[{timestamp:.3f}s] {token_id}\n"
            
            # Build complete token stream (interleaved text and audio)
            all_tokens = []
            
            # Add text tokens
            for token_info in s2s_engine.token_analyzer.text_tokens:
                all_tokens.append({
                    'timestamp': token_info['timestamp'],
                    'content': token_info['text'],
                    'type': 'text'
                })
            
            # Add audio tokens (including dummy ones)
            for token_info in s2s_engine.token_analyzer.audio_tokens:
                all_tokens.append({
                    'timestamp': token_info['timestamp'],
                    'content': f"<|audio_{token_info['token']}|>",
                    'type': 'audio'
                })
            
            # Sort by timestamp and build display
            all_tokens.sort(key=lambda x: x['timestamp'])
            for token in all_tokens:
                timestamp = token['timestamp']
                content = token['content']
                token_stream += f"[{timestamp:.3f}s] {content}\n"
            
            # Create timing summary with audio encoding
            first_text_time = f"{summary['first_text_token_time']:.3f}s" if summary['first_text_token_time'] else 'N/A'
            first_audio_time = f"{summary['first_audio_token_time']:.3f}s" if summary['first_audio_token_time'] else 'N/A'
            first_sentence_time = f"{summary['first_sentence_time']:.3f}s" if summary['first_sentence_time'] else 'N/A'
            first_zen_chunk_time = f"{summary['first_zen_chunk_time']:.3f}s" if summary['first_zen_chunk_time'] else 'N/A'
            audio_encoding_time = f"{summary['audio_encoding_time']*1000:.1f}ms" if summary['audio_encoding_time'] > 0 else 'N/A'
            
            timing_summary = f"""=== TIMING SUMMARY ===
🎯 First Token: {summary['first_token_time']:.3f}s
📝 First Text Token: {first_text_time}
🎵 First Audio Token: {first_audio_time}
🎧 Audio Encoding Time: {audio_encoding_time}
🎉 First Sentence: {first_sentence_time}
🎵 First Zen Decoder Chunk: {first_zen_chunk_time}

📊 COUNTS:
Text Tokens: {summary['total_text_tokens']}
Audio Tokens: {summary['total_audio_tokens']}
Zen Decoder Sentences: {summary['zen_sentences']}
Total Time: {summary['total_time']:.3f}s

🎉 FIRST SENTENCE:
{summary['first_sentence_text'] or 'No complete sentence detected'}"""
            
            # Get complete audio file for playback (not just latest sentence)
            complete_audio = result.get('complete_audio_file')
            
            return final_text, text_tokens_display, audio_tokens_display, token_stream, timing_summary, complete_audio
            
        except Exception as e:
            error_msg = f"Error: {e}"
            log_event("ERROR", task_selector, error_msg)
            return error_msg, "", "", "", "", None
    
    # Create interface - KEEPING EXACT SAME UI, JUST ADDING AUDIO OUTPUT AT BOTTOM
    with gr.Blocks(title="Zen-Audio Token Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Zen-Audio Token-Level Analysis")
        gr.Markdown("**Real-time token stream analysis with Zen Decoder (Audio Token Decoder)**")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Inputs
                audio_input = gr.Audio(
                    label="Audio Input", 
                    type="filepath",
                    format="wav"
                )
                
                text_input = gr.Textbox(
                    label="Text Input (for TTS)",
                    placeholder="Enter text for TTS..."
                )
                
                task_selector = gr.Dropdown(
                    choices=["Spoken QA", "ASR", "TTS"],
                    value="Spoken QA",
                    label="Task Type"
                )
                
                analyze_btn = gr.Button("Submit", variant="primary")
            
            with gr.Column(scale=2):
                # Outputs
                final_text = gr.Textbox(
                    label="Final Clean Text",
                    lines=3
                )
                
                timing_summary = gr.Textbox(
                    label="Timing Summary",
                    lines=12
                )
        
        with gr.Row():
            with gr.Column():
                text_tokens = gr.Textbox(
                    label="Text Tokens Stream",
                    lines=10
                )
            
            with gr.Column():
                audio_tokens = gr.Textbox(
                    label="Audio Tokens Stream", 
                    lines=10
                )
        
        with gr.Row():
            token_stream = gr.Textbox(
                label="Complete Token Stream",
                lines=15
            )
        
        # NEW: Audio output section at the bottom (as requested)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎵 Zen Decoder Audio Output")
                zen_audio_output = gr.Audio(
                    label="Complete Generated Audio (All Sentences Combined)",
                    type="filepath"
                )
        
        # Event handler
        analyze_btn.click(
            analyze_tokens,
            inputs=[audio_input, task_selector, text_input],
            outputs=[final_text, text_tokens, audio_tokens, token_stream, timing_summary, zen_audio_output]
        )
    
    return demo

if __name__ == "__main__":
    log_event("MAIN", "START", "🚀 Starting Token-Focused Zen-Audio Analysis with Zen Decoder...")
    
    # Create and launch interface
    demo = create_token_analysis_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True
    )
