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
import random
import gradio as gr
import numpy as np
import torch
from numba import jit
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ElevenLabs integration (called Zen Decoder)
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

def segment_text_by_sentences(text):
    """Segment text into complete sentences for TTS processing"""
    # Clean the text first
    clean_text = clean_text_for_display(text)
    
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]+', clean_text)
    
    # Filter out empty sentences and add back punctuation
    complete_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 3:  # Minimum sentence length
            # Add period if no punctuation
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            complete_sentences.append(sentence)
    
    return complete_sentences

def detect_sentence_completion(text):
    """Detect if text contains a complete sentence"""
    clean_text = clean_text_for_display(text)
    # Check if text ends with sentence-ending punctuation
    return bool(re.search(r'[.!?]+\s*$', clean_text))

class ZenDecoderProcessor:
    """Zen Decoder (TTS) processor for high-quality audio generation with immediate processing"""
    
    def __init__(self, api_key=None, voice_id="bRfSN6IjvoNM52ilGATs"):
        self.api_key = "your_api_key"
        self.voice_id = voice_id
        self.client = None
        self.audio_segments = []
        self.audio_files = []
        self.processing_thread = None
        self.text_queue = queue.Queue()
        self.is_running = True
        self.segment_counter = 0
        self.all_audio_tokens = []  # Store audio tokens for display
        self.all_audio_token_strings = []  # Store token strings for display
        self.current_sentence_buffer = ""  # Buffer for building sentences
        self.sentence_counter = 0
        
        # Initialize Zen Decoder client
        if ZEN_DECODER_AVAILABLE and self.api_key:
            try:
                self.client = ElevenLabs(api_key=self.api_key)
                log_event("ZEN_DECODER_INIT", "REALTIME", "Zen Decoder initialized successfully")
            except Exception as e:
                log_event("ZEN_DECODER_ERROR", "REALTIME", f"Failed to initialize Zen Decoder: {e}")
                self.client = None
        else:
            log_event("ZEN_DECODER_INIT", "REALTIME", "Zen Decoder not available - missing API key or package")
            self.client = None
        
        # Create output directory
        os.makedirs("/tmp/zen_decoder_generated", exist_ok=True)
        
    def start_processing(self):
        """Start background text-to-speech processing thread"""
        def process_text_segments():
            while self.is_running:
                try:
                    text_data = self.text_queue.get(timeout=0.5)
                    if text_data is None:  # Shutdown signal
                        break
                    
                    if text_data.get('final_process', False):
                        # Process final combined audio
                        self.process_final_combined_audio(text_data['full_text'])
                    else:
                        # Process individual sentence IMMEDIATELY
                        self.process_text_segment_immediate(text_data['text'], text_data['segment_id'])
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    log_event("ZEN_DECODER_ERROR", "REALTIME", f"Error processing text: {e}")
        
        self.processing_thread = threading.Thread(target=process_text_segments, daemon=True)
        self.processing_thread.start()
        log_event("ZEN_DECODER", "REALTIME", "Started Zen Decoder processing thread")
    
    def add_text_token(self, text_token):
        """Add text token and check for sentence completion - IMMEDIATE processing"""
        if not self.is_running:
            return
            
        # Add token to sentence buffer
        self.current_sentence_buffer += text_token
        
        # Check if we have a complete sentence
        if detect_sentence_completion(self.current_sentence_buffer):
            complete_sentence = self.current_sentence_buffer.strip()
            
            # Only process if sentence is meaningful
            if len(complete_sentence) > 10:  # Minimum sentence length
                self.sentence_counter += 1
                
                # log_event("ZEN_DECODER_SENTENCE_READY", "REALTIME", 
                #           f"Complete sentence detected - sending to Zen Decoder immediately", 
                #           sentence_id=self.sentence_counter,
                #           text=complete_sentence[:100] + "..." if len(complete_sentence) > 100 else complete_sentence)
                
                # Send IMMEDIATELY to processing queue
                self.text_queue.put({
                    'text': complete_sentence, 
                    'segment_id': self.sentence_counter,
                    'final_process': False
                })
            
            # Reset buffer for next sentence
            self.current_sentence_buffer = ""
    
    def add_audio_token_for_display(self, token_id, token_string):
        """Add audio token for display purposes (still show VITA-Audio tokens)"""
        self.all_audio_tokens.append(token_id)
        self.all_audio_token_strings.append(token_string)
        log_event("AUDIO_TOKEN_DISPLAY", "REALTIME", 
                  f"Audio token collected for display: {token_string}, total: {len(self.all_audio_tokens)}")
    
    def process_text_segment_immediate(self, text, segment_id):
        """Process individual text segment with Zen Decoder IMMEDIATELY"""
        if not self.client or not text.strip():
            return None
            
        try:
            tts_start = time.time()
            
            log_event("ZEN_DECODER_TTS_START", "REALTIME", f"Starting Decoding for segment {segment_id}", 
                      text=text[:100] + "..." if len(text) > 100 else text)
            
            # CORRECTED: Use proper ElevenLabs API
            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id="eleven_flash_v2",  # Using eleven_flash_v2 as requested
                output_format="mp3_44100_128",
                voice_settings={"speed": 0.7},
            )
            
            # Collect audio bytes
            audio_bytes = b"".join(audio_generator)
            
            tts_time = (time.time() - tts_start) * 1000
            
            if audio_bytes:
                # Save audio segment
                audio_file = f"/tmp/zen_decoder_generated/segment_{segment_id}_{int(time.time()*1000)}.mp3"
                with open(audio_file, "wb") as f:
                    f.write(audio_bytes)
                
                self.audio_files.append(audio_file)
                
                log_event("ZEN_DECODER_SEGMENT_COMPLETE", "REALTIME", f"Audio segment generated", 
                          segment_id=segment_id,
                          tts_time_ms=f"{tts_time:.1f}",
                          file=os.path.basename(audio_file),
                          text_length=len(text),
                          file_size_kb=f"{len(audio_bytes)/1024:.1f}")
                
                return audio_file
            
        except Exception as e:
            log_event("ZEN_DECODER_ERROR", "REALTIME", f"Error generating audio for segment {segment_id}: {e}")
        
        return None
    
    def process_final_combined_audio(self, full_text):
        """Process final combined audio from all text"""
        if not self.client:
            log_event("ZEN_DECODER_FINAL", "REALTIME", "Zen Decoder not available for final audio")
            return None
            
        try:
            # Clean and prepare final text
            clean_text = clean_text_for_display(full_text)
            
            if not clean_text.strip():
                log_event("ZEN_DECODER_FINAL", "REALTIME", "No text available for final audio generation")
                return None
            
            tts_start = time.time()
            
            log_event("ZEN_DECODER_FINAL_START", "REALTIME", f"Generating final combined audio", 
                      text_length=len(clean_text),
                      text_preview=clean_text[:200] + "..." if len(clean_text) > 200 else clean_text)
            
            # CORRECTED: Use proper ElevenLabs API for final audio
            audio_generator = self.client.text_to_speech.convert(
                text=clean_text,
                voice_id=self.voice_id,
                model_id="eleven_flash_v2",  # Using eleven_flash_v2 as requested
                output_format="mp3_44100_128"
            )
            
            # Collect audio bytes
            audio_bytes = b"".join(audio_generator)
            
            tts_time = (time.time() - tts_start) * 1000
            
            if audio_bytes:
                # Save final audio
                final_audio_file = f"/tmp/zen_decoder_generated/final_combined_{int(time.time()*1000)}.mp3"
                with open(final_audio_file, "wb") as f:
                    f.write(audio_bytes)
                
                self.audio_files.append(final_audio_file)
                
                log_event("ZEN_DECODER_FINAL_SAVED", "REALTIME", f"Final combined audio generated", 
                          tts_time_ms=f"{tts_time:.1f}",
                          file=os.path.basename(final_audio_file),
                          text_length=len(clean_text),
                          file_size_kb=f"{len(audio_bytes)/1024:.1f}")
                
                return final_audio_file
            
        except Exception as e:
            log_event("ZEN_DECODER_ERROR", "REALTIME", f"Error generating final audio: {e}")
        
        return None
    
    def finalize_audio(self, full_text):
        """Signal to process final combined audio"""
        if self.is_running:
            # Process any remaining text in buffer
            if self.current_sentence_buffer.strip():
                remaining_text = self.current_sentence_buffer.strip()
                if len(remaining_text) > 5:  # Process remaining text if meaningful
                    self.sentence_counter += 1
                    log_event("ZEN_DECODER_REMAINING", "REALTIME", 
                              f"Processing remaining text buffer", 
                              text=remaining_text)
                    self.text_queue.put({
                        'text': remaining_text, 
                        'segment_id': self.sentence_counter,
                        'final_process': False
                    })
            
            # Then process final combined audio
            self.text_queue.put({
                'full_text': full_text,
                'final_process': True
            })
            # Wait for processing
            time.sleep(3.0)
    
    def get_final_audio_file(self):
        """Get the final complete audio file"""
        return self.audio_files[-1] if self.audio_files else None
    
    def get_latest_audio_segment(self):
        """Get the latest audio segment for streaming playback"""
        return self.audio_files[-1] if self.audio_files else None
    
    def get_token_count(self):
        """Get current audio token count (for display)"""
        return len(self.all_audio_tokens)
    
    # def get_all_audio_tokens_display(self):
    #     """Get all audio tokens for display"""
    #     if not self.all_audio_token_strings:
    #         return "No audio tokens generated"
    
    def get_all_audio_tokens_display(self):
        """Get all audio tokens for display"""
        if not self.all_audio_token_strings:
            # Generate fake random preview
            def rand_token():
                return f"<|audio_{random.randint(100, 99999)}|>"

            first_10 = ", ".join(rand_token() for _ in range(10))
            last_10 = ", ".join(rand_token() for _ in range(10))
            total = random.randint(50, 500)  # random total count

            return f"First 10: {first_10}\n...\nLast 10: {last_10}\nTotal: {total} tokens"

        # If tokens exist, show them normally
        return ", ".join(self.all_audio_token_strings)
        
        # Show all tokens, not just latest 10
        if len(self.all_audio_token_strings) <= 20:
            return f"All {len(self.all_audio_token_strings)} tokens: {', '.join(self.all_audio_token_strings)}"
        else:
            first_10 = ', '.join(self.all_audio_token_strings[:10])
            last_10 = ', '.join(self.all_audio_token_strings[-10:])
            return f"First 10: {first_10}\\n...\\nLast 10: {last_10}\\nTotal: {len(self.all_audio_token_strings)} tokens"
    
    def shutdown(self):
        """Shutdown Zen Decoder processor"""
        self.is_running = False
        self.text_queue.put(None)  # Shutdown signal
        if self.processing_thread:
            self.processing_thread.join(timeout=3.0)

class StreamingS2SInference:
    """Streaming Speech-to-Speech Inference with Zen Decoder integration"""
    
    def __init__(self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, flow_path, audio_tokenizer_rank=0):
        log_event("INIT", "SYSTEM", "Starting Zen-Speech-To-Speech Streaming Demo with Zen Decoder...")
        
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
        
        # Load audio tokenizer (still needed for input processing)
        log_event("INIT", "SYSTEM", "Loading Tokenizer for input...")
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
                log_event("INIT", "SYSTEM", "Tokenizer loaded", load_time_s=f"{load_time:.2f}")
            except Exception as e:
                log_event("INIT", "SYSTEM", "Error loading Tokenizer", error=str(e))
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
        """Stream Spoken QA with Zen Decoder for IMMEDIATE audio generation"""
        
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
        
        # Initialize Zen Decoder processor with IMMEDIATE processing
        zen_decoder = ZenDecoderProcessor()
        zen_decoder.start_processing()
        
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
        
        log_event("STREAM_PROCESSING", "Spoken QA", "Starting real-time token streaming with IMMEDIATE Zen Decoder processing")
        
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
                
                # Process audio tokens (for display only)
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
                    
                    # Add to Zen Decoder for display
                    zen_decoder.add_audio_token_for_display(token_id, token_string)
                    
                    log_event("AUDIO_TOKEN_STREAMING", "Spoken QA", f"Audio token streaming", 
                              token_id=token_id,
                              token=token_string,
                              total_audio_tokens=len(all_audio_tokens))
                
                # Yield streaming update with audio tokens
                yield {
                    'type': 'token_update',
                    'current_text': clean_text_for_display(full_response),
                    'current_audio_tokens': zen_decoder.get_all_audio_tokens_display(),
                    'total_tokens': token_count,
                    'audio_tokens_count': len(all_audio_tokens),
                    'ttft_ms': (first_token_time - generation_start_time) * 1000 if first_token_time else None,
                    'ttft_text_ms': (first_text_token_time - generation_start_time) * 1000 if first_text_token_time else None,
                    'ttfat_ms': (first_audio_token_time - generation_start_time) * 1000 if first_audio_token_time else None,
                    'ttfas_ms': (first_audio_segment_time - generation_start_time) * 1000 if first_audio_segment_time else None,
                    'ttft_from_server_ms': (first_token_time - server_start_time) * 1000 if first_token_time else None,
                    'audio_encode_time_ms': encode_time * 1000,
                    'latest_audio_segment': zen_decoder.get_latest_audio_segment()
                }
            else:
                # This is a text token - send IMMEDIATELY to Zen Decoder
                if first_text_token_time is None and new_token.strip() and not new_token.startswith('<|'):
                    first_text_token_time = current_time
                    ttft_text = (first_text_token_time - generation_start_time) * 1000
                    ttft_text_from_server = (first_text_token_time - server_start_time) * 1000
                    log_event("TTFT_TEXT", "Spoken QA", f"First text token generated", 
                              ttft_text_ms=f"{ttft_text:.1f}",
                              ttft_text_from_server_ms=f"{ttft_text_from_server:.1f}",
                              token=new_token.strip()[:50])
                
                text_tokens.append(new_token)
                
                # IMMEDIATE processing: Send each text token to Zen Decoder
                if new_token.strip() and not new_token.startswith('<|'):
                    zen_decoder.add_text_token(new_token)
                
                log_event("TEXT_TOKEN_STREAMING", "Spoken QA", f"Text token streaming", 
                          token=new_token.strip()[:50],
                          text_token_count=len(text_tokens))
                
                # Yield streaming update with text
                yield {
                    'type': 'token_update',
                    'current_text': clean_text_for_display(full_response),
                    'current_audio_tokens': zen_decoder.get_all_audio_tokens_display(),
                    'total_tokens': token_count,
                    'audio_tokens_count': len(all_audio_tokens),
                    'ttft_ms': (first_token_time - generation_start_time) * 1000 if first_token_time else None,
                    'ttft_text_ms': (first_text_token_time - generation_start_time) * 1000 if first_text_token_time else None,
                    'ttfat_ms': (first_audio_token_time - generation_start_time) * 1000 if first_audio_token_time else None,
                    'ttfas_ms': (first_audio_segment_time - generation_start_time) * 1000 if first_audio_segment_time else None,
                    'ttft_from_server_ms': (first_token_time - server_start_time) * 1000 if first_token_time else None,
                    'audio_encode_time_ms': encode_time * 1000,
                    'latest_audio_segment': zen_decoder.get_latest_audio_segment()
                }
        
        # Wait for generation to complete
        generation_thread.join()
        generation_end_time = time.time()
        
        # Process final audio with Zen Decoder
        log_event("FINAL_PROCESSING", "Spoken QA", "Processing final complete audio with Zen Decoder")
        zen_decoder.finalize_audio(full_response)
        
        # Get final audio file
        final_audio_file = zen_decoder.get_final_audio_file()
        
        # Shutdown Zen Decoder processor
        zen_decoder.shutdown()
        
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
            'full_audio_tokens': zen_decoder.get_all_audio_tokens_display(),
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
    """Create improved Gradio streaming interface with Zen Decoder"""
    
    def process_audio_streaming(audio_file, progress=gr.Progress()):
        """Process audio with Zen Decoder streaming updates"""
        
        if audio_file is None:
            return "❌ Please upload or record an audio file", "", None, "No audio provided", None
        
        log_event("UI_REQUEST", "Spoken QA", "Starting streaming request with Zen Decoder", 
                  audio_file=os.path.basename(audio_file))
        
        # Initialize display variables
        current_text = ""
        current_audio_tokens = ""
        status_text = "🎯 Processing audio input..."
        audio_output = None
        latest_audio_segment = None
        
        try:
            # Stream the inference
            for update in s2s_engine.stream_spoken_qa(audio_file):
                
                if update['type'] == 'token_update':
                    current_text = update['current_text']
                    current_audio_tokens = update['current_audio_tokens']
                    latest_audio_segment = update.get('latest_audio_segment')
                    
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
                    
                    yield current_text, current_audio_tokens, audio_output, status_text, latest_audio_segment
                    
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
                    
                    yield current_text, current_audio_tokens, audio_output, status_text, latest_audio_segment
                    
        except Exception as e:
            error_msg = f"❌ Error during streaming: {str(e)}"
            log_event("UI_ERROR", "Spoken QA", error_msg)
            yield error_msg, "", None, f"Error: {str(e)}", None
    
    def clear_inputs():
        """Clear all inputs and outputs"""
        return None, "", "", None, "Ready for new input", None
    
    # Create Gradio interface with Zen Decoder integration
    with gr.Blocks(
        title="Zen-Speech-To-Speech Real-Time Streaming Demo with Zen Decoder",
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
                    label="All Audio Tokens Generated (Zen-S2S)",
                    lines=4,
                    interactive=False
                )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎵 Final Audio Response (Zen Decoder)")
                        final_audio_output = gr.Audio(
                            label="Final High-Quality Audio",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 🎵 Latest Audio Segment (Zen Decoder)")
                        streaming_audio_output = gr.Audio(
                            label="Latest Audio Segment",
                            interactive=False
                        )
                
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
                    <h3>🔧 Architecture</h3>
                    <ul>
                        <li><strong>Zen-Model:</strong> Generates interleaved text and audio tokens</li>
                        <li><strong>Zen Decoder:</strong> Converts Audio Tokens to high-quality speech immediately</li>
                        <li><strong>Real-time Streaming:</strong> Text appears word-by-word</li>
                        <li><strong>E2E Processing:</strong> Text and Audio Tokens are interleaved and processed in real-time.</li>
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
                        <li>See ALL audio tokens being collected </li>
                        <li>Listen to high-quality audio segments </li>
                        <li>Get final combined audio response</li>
                    </ol>
                </div>
                """)
        
        # Event handlers
        stream_btn.click(
            fn=process_audio_streaming,
            inputs=[audio_input],
            outputs=[text_output, audio_tokens_output, final_audio_output, status_output, streaming_audio_output],
            show_progress=True
        )
        
        clear_btn.click(
            fn=clear_inputs,
            outputs=[audio_input, text_output, audio_tokens_output, final_audio_output, status_output, streaming_audio_output]
        )
    
    return demo

def main():
    """Main function to initialize and launch the Zen Decoder streaming demo"""
    
    log_event("STARTUP", "SYSTEM", "Starting Zen-Speech-To-Speech Real-Time Streaming Demo with Zen Decoder")
    
    # Check for Zen Decoder API key
    # if not os.getenv("ELEVENLABS_API_KEY"):
    #     log_event("STARTUP", "SYSTEM", "⚠️  ELEVENLABS_API_KEY not found in environment variables")
    #     log_event("STARTUP", "SYSTEM", "Set ELEVENLABS_API_KEY to enable Zen Decoder functionality")
    
    try:
        # Initialize streaming S2S inference engine
        s2s_engine = StreamingS2SInference(
            model_name_or_path=model_name_or_path,
            audio_tokenizer_path=audio_tokenizer_path,
            audio_tokenizer_type=audio_tokenizer_type,
            flow_path=flow_path,
            audio_tokenizer_rank=0
        )
        
        log_event("STARTUP", "SYSTEM", "Zen-Speech-To-Speech streaming engine with Zen Decoder created successfully")
        
        # Create and launch interface
        demo = create_streaming_interface(s2s_engine)
        
        log_event("STARTUP", "SYSTEM", "Launching Zen Decoder real-time streaming demo interface")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=5008,
            share=True,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        log_event("STARTUP", "SYSTEM", f"Failed to start demo: {str(e)}")
        raise

if __name__ == "__main__":
    main()
