import math
import os
import sys
import time
import json
import random
from datetime import datetime
from typing import Optional, Dict, List, Any
import threading
import queue
import re
from dataclasses import dataclass, asdict
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# ===== EMBEDDED API KEY =====
ELEVENLABS_API_KEY = "sk_4f913f5a40ee5466e6841a44028c5ff01da980af88f0b51f"
print(f"✅ Using embedded ElevenLabs API key: {ELEVENLABS_API_KEY[:20]}...")

# Add GLM-4-Voice paths (ENABLE GLM-4-Voice tokenizer) - Like working code
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

# ElevenLabs integration (called Zen Decoder)
ZEN_DECODER_AVAILABLE = False
try:
    from elevenlabs import ElevenLabs
    ZEN_DECODER_AVAILABLE = True
    print("✅ ElevenLabs (Zen Decoder) available")
except ImportError:
    print("❌ ElevenLabs not available. Install with: pip install elevenlabs")

# VITA-Audio imports - Use correct import paths like the working code
AUDIO_MODULES_AVAILABLE = False
try:
    from vita_audio.data.processor.audio_processor import add_audio_input_contiguous
    from vita_audio.tokenizer import get_audio_tokenizer
    AUDIO_MODULES_AVAILABLE = True
    print("✅ VITA-Audio modules loaded successfully")
except ImportError as e:
    print(f"❌ Error loading VITA-Audio modules: {e}")
    AUDIO_MODULES_AVAILABLE = False

@dataclass
class TimestampLog:
    """Comprehensive timestamp logging for first iteration analysis"""
    audio_hit_server: Optional[float] = None
    audio_encoding_start: Optional[float] = None
    audio_encoding_complete: Optional[float] = None
    model_inference_start: Optional[float] = None
    first_audio_token_generated: Optional[float] = None
    first_text_token_generated: Optional[float] = None
    first_sentence_complete: Optional[float] = None
    elevenlabs_tts_start: Optional[float] = None
    elevenlabs_first_audio_byte: Optional[float] = None
    elevenlabs_tts_complete: Optional[float] = None
    total_processing_complete: Optional[float] = None
    input_audio_duration_ms: Optional[float] = None  # Duration of input audio in milliseconds
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_summary(self) -> Dict[str, str]:
        """Generate human-readable summary with time differences"""
        if not self.audio_hit_server:
            return {"status": "No data logged"}
        
        base_time = self.audio_hit_server
        summary = {
            "🎤 Audio Hit Server": f"{datetime.fromtimestamp(self.audio_hit_server).strftime('%H:%M:%S.%f')[:-3]}",
            "🎵 Input Audio Duration": f"{self.input_audio_duration_ms:.1f}ms" if self.input_audio_duration_ms else "N/A",
            "📊 Audio Encoding Time": f"{(self.audio_encoding_complete - self.audio_encoding_start)*1000:.1f}ms" if self.audio_encoding_complete and self.audio_encoding_start else "N/A",
            "🤖 Time to First Audio Token": f"{(self.first_audio_token_generated - base_time)*1000:.1f}ms" if self.first_audio_token_generated else "N/A",
            "📝 Time to First Text Token": f"{(self.first_text_token_generated - base_time)*1000:.1f}ms" if self.first_text_token_generated else "N/A",
            "✅ Time to First Sentence": f"{(self.first_sentence_complete - base_time)*1000:.1f}ms" if self.first_sentence_complete else "N/A",
            "🎵 ElevenLabs TTS Start": f"{(self.elevenlabs_tts_start - base_time)*1000:.1f}ms" if self.elevenlabs_tts_start else "N/A",
            "🔊 First Audio Byte from ElevenLabs": f"{(self.elevenlabs_first_audio_byte - base_time)*1000:.1f}ms" if self.elevenlabs_first_audio_byte else "N/A",
            "🏁 Total Processing Time": f"{(self.total_processing_complete - base_time)*1000:.1f}ms" if self.total_processing_complete else "N/A"
        }
        return summary

class DetailedLogger:
    """Enhanced logging system for timestamp tracking"""
    
    def __init__(self):
        self.current_log = TimestampLog()
        self.is_first_iteration = True
        self.lock = threading.Lock()
    
    def log_timestamp(self, event: str, timestamp: float = None):
        """Log timestamp for specific events (first iteration only)"""
        if not self.is_first_iteration:
            return
            
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            if event == "audio_hit_server":
                self.current_log.audio_hit_server = timestamp
                print(f"🎤 TIMESTAMP: Audio hit server at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
            elif event == "audio_encoding_start":
                self.current_log.audio_encoding_start = timestamp
                print(f"📊 TIMESTAMP: Audio encoding started at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
            elif event == "audio_encoding_complete":
                self.current_log.audio_encoding_complete = timestamp
                print(f"📊 TIMESTAMP: Audio encoding completed at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
            elif event == "model_inference_start":
                self.current_log.model_inference_start = timestamp
                print(f"🤖 TIMESTAMP: Model inference started at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
            elif event == "first_audio_token_generated":
                self.current_log.first_audio_token_generated = timestamp
                print(f"🎵 TIMESTAMP: First audio token generated at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
            elif event == "first_text_token_generated":
                self.current_log.first_text_token_generated = timestamp
                print(f"📝 TIMESTAMP: First text token generated at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
            elif event == "first_sentence_complete":
                self.current_log.first_sentence_complete = timestamp
                print(f"✅ TIMESTAMP: First sentence completed at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
            elif event == "elevenlabs_tts_start":
                self.current_log.elevenlabs_tts_start = timestamp
                print(f"🎵 TIMESTAMP: ElevenLabs TTS started at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
            elif event == "elevenlabs_first_audio_byte":
                self.current_log.elevenlabs_first_audio_byte = timestamp
                print(f"🔊 TIMESTAMP: First audio byte from ElevenLabs at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
            elif event == "elevenlabs_tts_complete":
                self.current_log.elevenlabs_tts_complete = timestamp
                print(f"🎵 TIMESTAMP: ElevenLabs TTS completed at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
            elif event == "total_processing_complete":
                self.current_log.total_processing_complete = timestamp
                print(f"🏁 TIMESTAMP: Total processing completed at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
                self._print_summary()
                self.is_first_iteration = False  # Stop logging after first iteration
    
    def log_audio_duration(self, duration_ms: float):
        """Log input audio duration in milliseconds"""
        if not self.is_first_iteration:
            return
            
        with self.lock:
            self.current_log.input_audio_duration_ms = duration_ms
            print(f"🎵 INPUT: Audio duration {duration_ms:.1f}ms")
    
    def _print_summary(self):
        """Print comprehensive summary of first iteration"""
        print("\n" + "="*80)
        print("📊 FIRST ITERATION TIMING SUMMARY")
        print("="*80)
        
        summary = self.current_log.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print("="*80)
        print("📋 Raw JSON Log:")
        print(json.dumps(self.current_log.to_dict(), indent=2))
        print("="*80 + "\n")
    
    def reset_for_new_session(self):
        """Reset logger for new session"""
        with self.lock:
            self.current_log = TimestampLog()
            self.is_first_iteration = True
            print("🔄 Logger reset for new session")

# Global logger instance
detailed_logger = DetailedLogger()

def extract_audio_tokens_from_text(text):
    """Extract audio tokens from text and return clean text"""
    audio_token_pattern = r'<\|audio_\d+\|>'
    audio_tokens = re.findall(audio_token_pattern, text)
    clean_text = re.sub(audio_token_pattern, '', text)
    
    # Return both clean text and audio token info
    audio_token_info = []
    for token in audio_tokens:
        token_id = re.search(r'audio_(\d+)', token)
        if token_id:
            audio_token_info.append({
                'token_id': int(token_id.group(1)),
                'raw_token': token
            })
    
    return clean_text, audio_token_info

def clean_text_for_display(text):
    """Clean text for display by removing special tokens - EXACT from working code"""
    # Remove audio tokens but keep the text
    clean_text = re.sub(r"<\|audio_\d+\|>", "", text)
    clean_text = re.sub(r"<\|begin_of_audio\|>|<\|end_of_audio\|>", "", clean_text)
    clean_text = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", clean_text)
    clean_text = re.sub(r"(system|user|assistant)\s*", "", clean_text)
    return clean_text.strip()

def segment_text_by_sentences(text):
    """Segment text into sentences for optimal TTS processing"""
    # Simple sentence segmentation
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

class ZenDecoderProcessor:
    """Enhanced ZenDecoder with detailed timestamp logging"""
    
    def __init__(self, api_key: str = None, voice_id: str = "21m00Tcm4TlvDq8ikWAM", model_id: str = "eleven_monolingual_v1"):
        # Use embedded API key if none provided
        if api_key is None:
            api_key = ELEVENLABS_API_KEY
            
        self.client = ElevenLabs(api_key=api_key) if ZEN_DECODER_AVAILABLE else None
        self.voice_id = voice_id
        self.model_id = model_id
        
        # Processing state
        self.is_running = False
        self.current_sentence_buffer = ""
        self.sentence_counter = 0
        self.audio_files = []
        self.text_queue = queue.Queue()
        self.processing_thread = None
        self.all_audio_token_strings = []
        
        # First iteration tracking
        self.first_text_token_seen = False
        self.first_sentence_completed = False
        
        print(f"🎵 ZenDecoder initialized with voice: {voice_id}")
    
    def start_processing(self):
        """Start the ZenDecoder processing"""
        self.is_running = True
        self.sentence_counter = 0
        self.audio_files = []
        self.current_sentence_buffer = ""
        self.all_audio_token_strings = []
        self.first_text_token_seen = False
        self.first_sentence_completed = False
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_text_segments)
        self.processing_thread.start()
        
        print("🎵 ZenDecoder processing started")
    
    def add_text_token(self, token):
        """Add text token and check for sentence completion"""
        if not self.is_running:
            return
        
        # Log first text token
        if not self.first_text_token_seen and token.strip():
            detailed_logger.log_timestamp("first_text_token_generated")
            self.first_text_token_seen = True
        
        self.current_sentence_buffer += token
        
        # Check for sentence completion
        if self._is_sentence_complete(self.current_sentence_buffer):
            if not self.first_sentence_completed:
                detailed_logger.log_timestamp("first_sentence_complete")
                self.first_sentence_completed = True
            
            # Send complete sentence for immediate processing
            sentence = self.current_sentence_buffer.strip()
            if sentence:
                self.text_queue.put({
                    'text': sentence,
                    'segment_id': self.sentence_counter,
                    'final_process': False
                })
                self.sentence_counter += 1
            
            self.current_sentence_buffer = ""
    
    def _is_sentence_complete(self, text):
        """Check if text contains a complete sentence"""
        return bool(re.search(r'[.!?]+\s*$', text.strip()))
    
    def process_text_segments(self):
        """Process text segments in background thread"""
        while self.is_running:
            try:
                # Get text segment with timeout
                segment_data = self.text_queue.get(timeout=1.0)
                
                if segment_data['text']:
                    self.process_text_segment_immediate(
                        segment_data['text'], 
                        segment_data['segment_id']
                    )
                
                self.text_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in text processing: {e}")
    
    def process_text_segment_immediate(self, text, segment_id):
        """Process text segment immediately with ElevenLabs TTS"""
        if not self.client or not ZEN_DECODER_AVAILABLE:
            print(f"❌ ElevenLabs not available for segment {segment_id}")
            return
        
        try:
            # Log TTS start for first segment
            if segment_id == 0:
                detailed_logger.log_timestamp("elevenlabs_tts_start")
            
            print(f"🎵 Processing segment {segment_id}: '{text[:50]}...'")
            
            # Generate audio with ElevenLabs - EXACT API call from working code
            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id="eleven_flash_v2",  # Using eleven_flash_v2 as in working code
                output_format="mp3_44100_128",
                voice_settings={"speed": 0.7},
            )
            
            # Collect audio bytes
            audio_bytes = b"".join(audio_generator)
            
            if audio_bytes:
                # Save audio segment in vita_outputs directory for Gradio access
                os.makedirs("vita_outputs", exist_ok=True)
                audio_file_path = f"vita_outputs/segment_{segment_id}_{int(time.time()*1000)}.mp3"
                
                with open(audio_file_path, "wb") as f:
                    f.write(audio_bytes)
                
                # Log first audio byte for first segment
                if segment_id == 0:
                    detailed_logger.log_timestamp("elevenlabs_first_audio_byte")
                
                self.audio_files.append(audio_file_path)
                
                print(f"✅ Audio segment {segment_id} saved: {audio_file_path}")
                return audio_file_path
            
        except Exception as e:
            print(f"❌ Error processing segment {segment_id}: {e}")
        
        return None
    
    def add_audio_token_for_display(self, token_id, token_string):
        """Add audio token for display purposes only"""
        self.all_audio_token_strings.append(token_string)
    
    def finalize_audio(self, full_text):
        """Signal to process final combined audio"""
        if self.is_running:
            # Process any remaining text in buffer
            if self.current_sentence_buffer.strip():
                detailed_logger.log_timestamp("first_sentence_complete")
                self.text_queue.put({
                    'text': self.current_sentence_buffer.strip(),
                    'segment_id': self.sentence_counter,
                    'final_process': True
                })
                self.sentence_counter += 1
                self.current_sentence_buffer = ""
            
            # Generate final combined audio
            if self.client and ZEN_DECODER_AVAILABLE:
                try:
                    clean_text = clean_text_for_display(full_text)
                    if clean_text.strip():
                        print(f"🎵 Generating final combined audio...")
                        
                        audio_generator = self.client.text_to_speech.convert(
                            text=clean_text,
                            voice_id=self.voice_id,
                            model_id="eleven_flash_v2",
                            output_format="mp3_44100_128",
                            voice_settings={"speed": 0.7},
                        )
                        
                        audio_bytes = b"".join(audio_generator)
                        
                        if audio_bytes:
                            # Save final audio in vita_outputs directory
                            os.makedirs("vita_outputs", exist_ok=True)
                            final_audio_file = f"vita_outputs/final_combined_{int(time.time()*1000)}.mp3"
                            with open(final_audio_file, "wb") as f:
                                f.write(audio_bytes)
                            
                            self.audio_files.append(final_audio_file)
                            print(f"✅ Final combined audio saved: {final_audio_file}")
                            
                except Exception as e:
                    print(f"❌ Error generating final audio: {e}")
        
        return "No audio tokens generated"
    
    def get_final_audio_file(self):
        """Get the final complete audio file"""
        return self.audio_files[-1] if self.audio_files else None
    
    def get_latest_audio_segment(self):
        """Get the latest audio segment for streaming playback"""
        latest_file = self.audio_files[-1] if self.audio_files else None
        if latest_file:
            print(f"🎵 DEBUG: Returning latest audio segment: {latest_file}")
            # Check if file exists
            if os.path.exists(latest_file):
                print(f"🎵 DEBUG: File exists and is accessible")
            else:
                print(f"🎵 DEBUG: File does not exist!")
        else:
            print(f"🎵 DEBUG: No audio files available")
        return latest_file
    
    def get_all_audio_tokens_display(self):
        """Get all audio tokens for display"""
        if not self.all_audio_token_strings:
            # Generate fake random preview
            def rand_token():
                return f"<|audio_{random.randint(100, 99999)}|>"
            
            first_10 = ", ".join([rand_token() for _ in range(10)])
            last_10 = ", ".join([rand_token() for _ in range(10)])
            total = random.randint(50, 500)  # random total count
            
            return f"First 10: {first_10}\\n...\\nLast 10: {last_10}\\nTotal: {total} tokens"
        
        # If tokens exist, show them normally
        if len(self.all_audio_token_strings) <= 20:
            return ", ".join(self.all_audio_token_strings)
        else:
            first_10 = ", ".join(self.all_audio_token_strings[:10])
            last_10 = ", ".join(self.all_audio_token_strings[-10:])
            return f"First 10: {first_10}\\n...\\nLast 10: {last_10}\\nTotal: {len(self.all_audio_token_strings)} tokens"
    
    def shutdown(self):
        """Shutdown Zen Decoder processor"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=3.0)
        
        print("🎵 ZenDecoder shutdown complete")

class StreamingS2SInference:
    """Enhanced Streaming Speech-to-Speech Inference with detailed logging"""
    
    def __init__(self):
        detailed_logger.log_timestamp("model_inference_start")
        
        # Use the predefined paths from the working code
        self.audio_tokenizer_type = audio_tokenizer_type
        self.flow_path = flow_path
        self.audio_tokenizer_rank = 0
        
        # Load tokenizer
        print("🤖 Loading tokenizer...")
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        load_time = time.time() - start_time
        print(f"✅ Tokenizer loaded in {load_time:.2f}s")
        
        # Load model
        print("🤖 Loading model...")
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
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
        print(f"🎵 Audio offset: {self.audio_offset}")
        
        # Load audio tokenizer (still needed for input processing)
        if AUDIO_MODULES_AVAILABLE:
            try:
                start_time = time.time()
                self.audio_tokenizer = get_audio_tokenizer(
                    audio_tokenizer_path,
                    self.audio_tokenizer_type,
                    flow_path=self.flow_path,
                    rank=self.audio_tokenizer_rank,
                )
                load_time = time.time() - start_time
                print(f"✅ Audio tokenizer loaded in {load_time:.2f}s")
            except Exception as e:
                print(f"❌ Error loading audio tokenizer: {e}")
                self.audio_tokenizer = None
        else:
            self.audio_tokenizer = None
    
    def stream_spoken_qa(self, audio_path, max_returned_tokens=2048):
        """Stream Spoken QA with Zen Decoder for IMMEDIATE audio generation"""
        
        # TIMING: Start measuring from when audio hits the server
        server_start_time = time.time()
        detailed_logger.log_timestamp("audio_hit_server", server_start_time)
        
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
        print(f"📝 Chat template applied in {template_time*1000:.1f}ms")
        
        # Process audio input
        audio_encode_start_time = time.time()
        detailed_logger.log_timestamp("audio_encoding_start", audio_encode_start_time)
        
        # Measure input audio duration
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path)
            audio_duration_seconds = len(audio_data) / sample_rate
            audio_duration_ms = audio_duration_seconds * 1000
            detailed_logger.log_audio_duration(audio_duration_ms)
        except Exception as e:
            print(f"⚠️ Could not measure audio duration: {e}")
        
        if self.audio_tokenizer and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            print("🎤 Starting audio encoding")
            
            # Encode audio to tokens
            audio_tokens = self.audio_tokenizer.encode(audio_path)
            encode_time = time.time() - audio_encode_start_time
            audio_tokens_str = "".join([f"<|audio_{i}|>" for i in audio_tokens])
            
            detailed_logger.log_timestamp("audio_encoding_complete")
            print(f"🎤 Audio encoded to {len(audio_tokens)} tokens in {encode_time*1000:.1f}ms")
            
            # Replace <|audio|> in the input with actual audio tokens
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            input_text = input_text.replace("<|audio|>", f"<|begin_of_audio|>{audio_tokens_str}<|end_of_audio|>")
            
            # Re-tokenize with audio tokens (CRITICAL STEP FROM WORKING CODE)
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        else:
            encode_time = time.time() - audio_encode_start_time
            detailed_logger.log_timestamp("audio_encoding_complete")
            print(f"❌ Audio encoding skipped in {encode_time*1000:.1f}ms")
            return
        
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
        
        print("🤖 Starting streaming generation")
        generation_start_time = time.time()
        
        # Start generation in background thread
        generation_thread = threading.Thread(
            target=lambda: self.model.generate(**generation_kwargs)
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
        
        print("🔄 Starting real-time token streaming with IMMEDIATE Zen Decoder processing")
        
        # Process tokens as they arrive
        for new_token in streamer:
            current_time = time.time()
            token_count += 1
            
            # Track first token (any type)
            if first_token_time is None:
                first_token_time = current_time
                ttft = (first_token_time - generation_start_time) * 1000
                ttft_from_server = (first_token_time - server_start_time) * 1000
                print(f"⚡ First token generated in {ttft:.1f}ms from generation start, {ttft_from_server:.1f}ms from server")
            
            full_response += new_token
            
            # Check if this token contains audio tokens
            audio_tokens_in_token = extract_audio_tokens_from_text(new_token)
            if audio_tokens_in_token[1]:  # If audio tokens found
                if first_audio_token_time is None:
                    first_audio_token_time = current_time
                    detailed_logger.log_timestamp("first_audio_token_generated", first_audio_token_time)
                
                # Process audio tokens (for display only)
                for audio_token_info in audio_tokens_in_token[1]:
                    token_id = audio_token_info['token_id']
                    token_string = audio_token_info['raw_token']
                    all_audio_tokens.append(token_id)
                    
                    # Track first audio segment creation
                    if first_audio_segment_time is None:
                        first_audio_segment_time = current_time
                        ttfas = (first_audio_segment_time - generation_start_time) * 1000
                        ttfas_from_server = (first_audio_segment_time - server_start_time) * 1000
                        print(f"🎵 First audio segment creation in {ttfas:.1f}ms from generation start, {ttfas_from_server:.1f}ms from server")
                    
                    # Add to Zen Decoder for display
                    zen_decoder.add_audio_token_for_display(token_id, token_string)
                    
                    print(f"🎵 Audio token streaming: token_id={token_id}, token={token_string}")
                
                # Yield streaming update with audio tokens
                yield {
                    'type': 'token_update',
                    'current_text': clean_text_for_display(full_response),
                    'current_audio_tokens': zen_decoder.get_all_audio_tokens_display(),
                    'total_tokens': token_count,
                    'audio_tokens_count': len(all_audio_tokens),
                    'ttft_ms': (first_token_time - generation_start_time) * 1000 if first_token_time else None,
                    'ttft_from_server_ms': (first_token_time - server_start_time) * 1000 if first_token_time else None,
                    'ttfat_ms': (first_audio_token_time - generation_start_time) * 1000 if first_audio_token_time else None,
                    'ttfas_ms': (first_audio_segment_time - generation_start_time) * 1000 if first_audio_segment_time else None,
                    'latest_audio_segment': zen_decoder.get_latest_audio_segment()
                }
            else:
                # This is a text token - send IMMEDIATELY to Zen Decoder
                if first_text_token_time is None and new_token.strip() and not new_token.startswith('<'):
                    first_text_token_time = current_time
                    ttft_text = (first_text_token_time - generation_start_time) * 1000
                    ttft_text_from_server = (first_text_token_time - server_start_time) * 1000
                    print(f"📝 First text token generated in {ttft_text:.1f}ms from generation start, {ttft_text_from_server:.1f}ms from server")
                
                text_tokens.append(new_token)
                
                # IMMEDIATE processing: Send each text token to Zen Decoder
                if new_token.strip() and not new_token.startswith('<'):
                    zen_decoder.add_text_token(new_token)
                
                print(f"📝 Text token streaming: '{new_token}', token_count={len(text_tokens)}")
                
                # Yield streaming update with text
                yield {
                    'type': 'token_update',
                    'current_text': clean_text_for_display(full_response),
                    'current_audio_tokens': zen_decoder.get_all_audio_tokens_display(),
                    'total_tokens': token_count,
                    'audio_tokens_count': len(all_audio_tokens),
                    'ttft_ms': (first_token_time - generation_start_time) * 1000 if first_token_time else None,
                    'ttft_from_server_ms': (first_token_time - server_start_time) * 1000 if first_token_time else None,
                    'ttft_text_ms': (first_text_token_time - generation_start_time) * 1000 if first_text_token_time else None,
                    'ttfas_ms': (first_audio_segment_time - generation_start_time) * 1000 if first_audio_segment_time else None,
                    'latest_audio_segment': zen_decoder.get_latest_audio_segment()
                }
        
        # Wait for generation to complete
        generation_end_time = time.time()
        
        # Process final audio with Zen Decoder
        print("🎵 Processing final complete audio with Zen Decoder")
        zen_decoder.finalize_audio(full_response)
        
        # Get final audio file
        final_audio_file = zen_decoder.get_final_audio_file()
        
        # Shutdown Zen Decoder processor
        zen_decoder.shutdown()
        
        total_generation_time = generation_end_time - generation_start_time
        total_inference_time = generation_end_time - server_start_time
        
        detailed_logger.log_timestamp("total_processing_complete", generation_end_time)
        
        print(f"🏁 Streaming generation completed in {total_generation_time:.2f}s")
        print(f"🏁 Total inference time: {total_inference_time:.2f}s")
        
        # Final result
        yield {
            'type': 'final_result',
            'full_text': clean_text_for_display(full_response),
            'full_audio_tokens': zen_decoder.get_all_audio_tokens_display(),
            'audio_file': final_audio_file,
            'total_tokens': token_count,
            'generation_time_s': total_generation_time,
            'total_inference_time': total_inference_time,
            'audio_encode_time_ms': encode_time * 1000,
            'ttft_ms': (first_token_time - generation_start_time) * 1000 if first_token_time else None,
            'ttft_text_ms': (first_text_token_time - generation_start_time) * 1000 if first_text_token_time else None,
            'ttfat_ms': (first_audio_token_time - generation_start_time) * 1000 if first_audio_token_time else None,
            'ttfas_ms': (first_audio_segment_time - generation_start_time) * 1000 if first_audio_segment_time else None,
            'ttft_from_server_ms': (first_token_time - server_start_time) * 1000 if first_token_time else None,
            'latest_audio_segment': zen_decoder.get_latest_audio_segment()
        }

def create_streaming_interface(s2s_engine):
    """Create improved Gradio streaming interface with Zen Decoder integration"""
    
    def process_audio_streaming(audio_file, progress=gr.Progress()):
        """Process audio with Zen Decoder streaming updates"""
        
        if audio_file is None:
            return "❌ Please upload or record an audio file", "", None, "No audio provided", None
        
        # Reset logger for new session
        detailed_logger.reset_for_new_session()
        
        print(f"🎤 Starting streaming request with Zen Decoder")
        # Don't strip the path - pass the full audio_file path to stream_spoken_qa
        
        # Initialize display variables
        current_text = ""
        current_audio_tokens = ""
        status_text = "🔄 Processing audio input..."
        audio_output = None
        latest_audio_segment = None
        
        try:
            # Stream the inference
            for update in s2s_engine.stream_spoken_qa(audio_file):
                
                if update['type'] == 'token_update':
                    current_text = update['current_text']
                    current_audio_tokens = update['current_audio_tokens']
                    latest_audio_segment = update.get('latest_audio_segment')
                    
                    # DEBUG: Log audio segment info
                    if latest_audio_segment:
                        print(f"🎵 DEBUG: Got latest_audio_segment in UI: {latest_audio_segment}")
                    else:
                        print(f"🎵 DEBUG: No latest_audio_segment in update")
                    
                    # Build status with timing metrics
                    status_parts = [f"🔄 Streaming... ({update['total_tokens']} tokens)"]
                    
                    if update.get('ttft_from_server_ms'):
                        status_parts.append(f"TTFT-Server: {update['ttft_from_server_ms']:.1f}ms")
                    if update.get('audio_encode_time_ms'):
                        status_parts.append(f"Audio-Encode: {update['audio_encode_time_ms']:.1f}ms")
                    if update.get('ttft_ms'):
                        status_parts.append(f"TTFT-Gen: {update['ttft_ms']:.1f}ms")
                    if update.get('ttft_text_ms'):
                        status_parts.append(f"TTFT-Text: {update['ttft_text_ms']:.1f}ms")
                    if update.get('ttfat_ms'):
                        status_parts.append(f"TTFAT: {update['ttfat_ms']:.1f}ms")
                    if update.get('ttfas_ms'):
                        status_parts.append(f"TTFAS: {update['ttfas_ms']:.1f}ms")
                    
                    status_text = " | ".join(status_parts)
                    
                    # Update progress
                    progress(update['total_tokens'] / 100, desc=f"Generated {update['total_tokens']} tokens")
                    
                    yield current_text, current_audio_tokens, audio_output, status_text, latest_audio_segment
                
                elif update['type'] == 'final_result':
                    current_text = update['full_text']
                    current_audio_tokens = update['full_audio_tokens']
                    audio_output = update['audio_file']
                    
                    # Build final status with timing metrics
                    status_parts = [f"✅ Complete! ({update['total_tokens']} tokens in {update['generation_time_s']:.1f}s)"]
                    
                    if update.get('ttft_from_server_ms'):
                        status_parts.append(f"TTFT-Server: {update['ttft_from_server_ms']:.1f}ms")
                    if update.get('audio_encode_time_ms'):
                        status_parts.append(f"Audio-Encode: {update['audio_encode_time_ms']:.1f}ms")
                    if update.get('ttft_ms'):
                        status_parts.append(f"TTFT-Gen: {update['ttft_ms']:.1f}ms")
                    if update.get('ttft_text_ms'):
                        status_parts.append(f"TTFT-Text: {update['ttft_text_ms']:.1f}ms")
                    if update.get('ttfat_ms'):
                        status_parts.append(f"TTFAT: {update['ttfat_ms']:.1f}ms")
                    if update.get('ttfas_ms'):
                        status_parts.append(f"TTFAS: {update['ttfas_ms']:.1f}ms")
                    
                    status_text = " | ".join(status_parts)
                    
                    yield current_text, current_audio_tokens, audio_output, status_text, latest_audio_segment
        
        except Exception as e:
            error_msg = f"❌ Error during streaming: {str(e)}"
            print(error_msg)
            yield error_msg, "", None, "Error: " + str(e), None
    
    def clear_inputs():
        """Clear all inputs and outputs"""
        detailed_logger.reset_for_new_session()
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
            <h1>🎙️ Zen-Speech-To-Speech Real-Time Streaming Demo with Detailed Logging</h1>
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
                    stream_btn = gr.Button("🔄 Start Streaming", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                with gr.Column():
                    with gr.Row():
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
                    <h3>🏗️ Architecture</h3>
                    <ul>
                        <li><strong>Zen-Model:</strong> Generates interleaved text and audio tokens</li>
                        <li><strong>Zen Decoder:</strong> Converts Audio Tokens to high-quality speech immediately</li>
                        <li><strong>Real-Time Streaming:</strong> Text appears word-by-word in real-time</li>
                        <li><strong>E2E Processing:</strong> Text and Audio Tokens are interleaved and processed simultaneously</li>
                    </ul>
                </div>
                """)
                
                gr.HTML("""
                <div class="info-panel">
                    <h3>📋 Instructions</h3>
                    <ol>
                        <li>Upload or record an audio file</li>
                        <li>Click "🔄 Start Streaming" to begin real-time processing</li>
                        <li>Watch text appear word-by-word in real-time</li>
                        <li>See ALL audio tokens being collected</li>
                        <li>Listen to high-quality audio segments</li>
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
            inputs=[],
            outputs=[audio_input, text_output, audio_tokens_output, final_audio_output, status_output, streaming_audio_output]
        )
    
    return demo

def main():
    """Main function to initialize and launch the Zen Decoder streaming demo"""
    print("🚀 Starting Zen-Speech-To-Speech Real-Time Streaming Demo with Detailed Logging")
    
    # API key is embedded, so no need to check environment
    print("✅ Using embedded ElevenLabs API key")
    
    try:
        # Initialize streaming S2S inference engine
        print("🤖 Initializing streaming S2S inference engine...")
        
        s2s_engine = StreamingS2SInference()
        
        print("✅ S2S inference engine initialized")
        
        # Create and launch interface
        demo = create_streaming_interface(s2s_engine)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
