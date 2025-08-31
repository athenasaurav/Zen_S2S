import copy
import math
import os
import sys
import time
import warnings
import re
from datetime import datetime, timezone
from threading import Thread
from queue import Queue

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

def log_event(event_type, task_type, message, **kwargs):
    """Log event with UTC timestamp and task context"""
    timestamp = get_utc_timestamp()
    task_info = f"[{task_type}]" if task_type else ""
    print(f"{timestamp} {task_info} {event_type}: {message}")
    
    # Log additional kwargs if provided
    for key, value in kwargs.items():
        if value is not None:
            print(f"{timestamp} {task_info} {event_type}: {key}: {value}")

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

class StreamingAudioProcessor:
    """Handle real-time audio streaming and processing"""
    
    def __init__(self, audio_tokenizer, sample_rate=16000):
        self.audio_tokenizer = audio_tokenizer
        self.sample_rate = sample_rate
        self.audio_queue = Queue()
        self.is_streaming = False
        
    def start_streaming(self):
        """Start audio streaming mode"""
        self.is_streaming = True
        self.audio_queue = Queue()
        log_event("STREAM", "AUDIO", "Audio streaming started")
        
    def stop_streaming(self):
        """Stop audio streaming mode"""
        self.is_streaming = False
        log_event("STREAM", "AUDIO", "Audio streaming stopped")
        
    def process_audio_chunk(self, audio_tokens, chunk_id):
        """Process individual audio chunk for streaming"""
        if not self.is_streaming or not audio_tokens:
            return None
            
        try:
            chunk_start_time = time.time()
            audio_chunk = self.audio_tokenizer.decode(audio_tokens)
            chunk_time = time.time() - chunk_start_time
            
            log_event("STREAM", "AUDIO", f"Chunk {chunk_id}: {len(audio_tokens)} tokens → {audio_chunk.shape[0]/self.sample_rate:.2f}s audio in {chunk_time:.3f}s")
            
            # Add to queue for real-time playback
            self.audio_queue.put({
                'chunk_id': chunk_id,
                'audio': audio_chunk,
                'tokens': len(audio_tokens),
                'duration': audio_chunk.shape[0]/self.sample_rate,
                'processing_time': chunk_time
            })
            
            return audio_chunk
            
        except Exception as e:
            log_event("STREAM", "AUDIO", f"Error processing chunk {chunk_id}: {e}")
            return None

class S2SInferenceOptimized:
    """Optimized Speech-to-Speech Inference with Turbo mode and streaming"""
    
    def __init__(self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, flow_path, 
                 audio_tokenizer_rank=0, use_turbo=True):
        self.model_name_or_path = model_name_or_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.audio_tokenizer_type = audio_tokenizer_type
        self.flow_path = flow_path
        self.audio_tokenizer_rank = audio_tokenizer_rank
        self.use_turbo = use_turbo
        
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
                log_event("INIT", "SYSTEM", "Audio tokenizer loaded")
                
                # Initialize streaming processor
                self.streaming_processor = StreamingAudioProcessor(self.audio_tokenizer)
                
            except Exception as e:
                log_event("INIT", "SYSTEM", f"Error loading audio tokenizer: {e}")
                self.audio_tokenizer = None
                self.streaming_processor = None
        else:
            self.audio_tokenizer = None
            self.streaming_processor = None
            
        # Configure generation with TURBO mode
        self.configure_generation_mode()
        
        # Get audio offset
        self.audio_offset = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")
        log_event("INIT", "SYSTEM", f"Audio offset: {self.audio_offset}")
        
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
            log_event("INIT", "SYSTEM", "🚀 TURBO MODE enabled: [1, 10] - Expected ~295 tokens/second, ~3.4ms/token")
        else:
            # BOOST MODE: [1, 10, 4, 10] for balanced performance (170 tokens/second)
            self.model.generation_config.mtp_inference_mode = [1, 10, 4, 10]
            log_event("INIT", "SYSTEM", "⚡ BOOST MODE enabled: [1, 10, 4, 10] - Expected ~170 tokens/second, ~5.9ms/token")
            
        log_event("INIT", "SYSTEM", f"MTP inference mode: {self.model.generation_config.mtp_inference_mode}")
        
        if self.model.config.model_type == "hunyuan":
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
            
    def switch_mode(self, use_turbo):
        """Switch between Turbo and Boost modes dynamically"""
        self.use_turbo = use_turbo
        self.configure_generation_mode()
        
    def run_infer_streaming(self, audio_path=None, prompt_audio_path=None, message="", task_type="Spoken QA",
                           stream_stride=4, max_returned_tokens=4096, sample_rate=16000):
        """Streaming inference with real-time audio generation"""
        
        request_start_time = time.time()
        request_start_utc = get_utc_timestamp()
        
        log_event("STREAM_INFER", task_type, f"🎬 Starting streaming inference")
        log_event("STREAM_INFER", task_type, f"Mode: {'TURBO' if self.use_turbo else 'BOOST'}")
        log_event("STREAM_INFER", task_type, f"MTP config: {self.model.generation_config.mtp_inference_mode}")
        
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
        template_start_time = time.time()
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        template_time = time.time() - template_start_time
        log_event("STREAM_INFER", task_type, f"Chat template applied in {template_time:.3f}s")

        # Handle audio input processing - CRITICAL FIX: Use EXACT same approach as original working code
        audios = None
        audio_indices = None
        
        # CRITICAL: For ASR and Spoken QA with audio, we need to process the audio file
        if (audio_path is not None or prompt_audio_path is not None) and self.audio_tokenizer:
            log_event("STREAM_INFER", task_type, f"Processing audio input with tokenizer...")
            
            # Check if audio tokenizer applies to user role
            if self.audio_tokenizer.apply_to_role("user", is_contiguous=True):
                log_event("STREAM_INFER", task_type, "Using contiguous codec for audio processing")
                # Contiguous codec
                audio_paths = []
                if audio_path is not None:
                    audio_paths.append(audio_path)
                    log_event("STREAM_INFER", task_type, f"Added audio_path: {audio_path}")
                if prompt_audio_path is not None:
                    audio_paths.append(prompt_audio_path)
                    log_event("STREAM_INFER", task_type, f"Added prompt_audio_path: {prompt_audio_path}")
                    
                try:
                    audio_process_start = time.time()
                    # CRITICAL FIX: Use EXACT same signature as original working code
                    # Pass input_ids as tensor (not converted to list), audio_paths, tokenizer, audio_tokenizer
                    input_ids, audios, audio_indices = add_audio_input_contiguous(
                        input_ids, audio_paths, self.tokenizer, self.audio_tokenizer
                    )
                    audio_process_time = time.time() - audio_process_start
                    log_event("STREAM_INFER", task_type, f"Processed {len(audio_paths)} audio files in {audio_process_time:.3f}s")
                    log_event("STREAM_INFER", task_type, f"audios shape: {audios.shape if audios is not None else None}")
                    log_event("STREAM_INFER", task_type, f"audio_indices: {audio_indices}")
                    
                    # Log detailed contiguous codec processing
                    log_event("AUDIO_ENCODING", task_type, f"Contiguous codec processing breakdown:")
                    log_event("AUDIO_ENCODING", task_type, f"  - Audio files processed: {len(audio_paths)}")
                    log_event("AUDIO_ENCODING", task_type, f"  - Processing time: {audio_process_time:.3f}s")
                    log_event("AUDIO_ENCODING", task_type, f"  - Audio tensor shape: {audios.shape if audios is not None else 'None'}")
                    log_event("AUDIO_ENCODING", task_type, f"  - Audio indices: {audio_indices}")
                    if audio_paths:
                        for i, path in enumerate(audio_paths):
                            log_event("AUDIO_ENCODING", task_type, f"  - Audio file {i+1}: {os.path.basename(path)}")
                            
                except Exception as e:
                    log_event("STREAM_INFER", task_type, f"Audio processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    audios = None
                    audio_indices = None
                    
            elif self.audio_tokenizer.apply_to_role("user", is_discrete=True):
                log_event("STREAM_INFER", task_type, "Using discrete codec for audio processing")
                # Discrete codec - encode audio to tokens
                if audio_path is not None:
                    try:
                        audio_encode_start = time.time()
                        audio_tokens = self.audio_tokenizer.encode(audio_path)
                        audio_encode_time = time.time() - audio_encode_start
                        audio_tokens_str = "".join([f"<|audio_{i}|>" for i in audio_tokens])
                        log_event("STREAM_INFER", task_type, f"Encoded audio to {len(audio_tokens)} tokens in {audio_encode_time:.3f}s")
                        
                        # Log detailed audio encoding analysis
                        log_event("AUDIO_ENCODING", task_type, f"Audio input processing breakdown:")
                        log_event("AUDIO_ENCODING", task_type, f"  - Audio file: {os.path.basename(audio_path)}")
                        log_event("AUDIO_ENCODING", task_type, f"  - Audio encoding time: {audio_encode_time:.3f}s")
                        log_event("AUDIO_ENCODING", task_type, f"  - Generated {len(audio_tokens)} audio tokens")
                        log_event("AUDIO_ENCODING", task_type, f"  - First audio token ID: {audio_tokens[0] if audio_tokens else 'None'}")
                        log_event("AUDIO_ENCODING", task_type, f"  - Last audio token ID: {audio_tokens[-1] if audio_tokens else 'None'}")
                        log_event("AUDIO_ENCODING", task_type, f"  - Audio token range: {min(audio_tokens) if audio_tokens else 'None'} to {max(audio_tokens) if audio_tokens else 'None'}")
                        
                        # Replace <|audio|> in the input with actual audio tokens
                        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                        input_text = input_text.replace("<|audio|>", f"<|begin_of_audio|>{audio_tokens_str}<|end_of_audio|>")
                        
                        # Re-tokenize with audio tokens
                        retokenize_start = time.time()
                        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
                        retokenize_time = time.time() - retokenize_start
                        log_event("STREAM_INFER", task_type, f"Re-tokenized input with audio tokens in {retokenize_time:.3f}s")
                        
                        # Log total audio processing time
                        total_audio_process_time = audio_encode_time + retokenize_time
                        log_event("AUDIO_ENCODING", task_type, f"Total audio processing time: {total_audio_process_time:.3f}s (encode: {audio_encode_time:.3f}s + retokenize: {retokenize_time:.3f}s)")
                        
                    except Exception as e:
                        log_event("STREAM_INFER", task_type, f"Discrete audio processing error: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                log_event("STREAM_INFER", task_type, "Audio tokenizer doesn't apply to user role")

        # Move to device
        device_move_start = time.time()
        input_ids = input_ids.to(self.model.device)
        device_move_time = time.time() - device_move_start
        log_event("STREAM_INFER", task_type, f"Input moved to device in {device_move_time:.3f}s")

        # Start streaming processor
        if self.streaming_processor:
            self.streaming_processor.start_streaming()

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
            generation_kwargs["audios"] = audios
        if audio_indices is not None:
            generation_kwargs["audio_indices"] = audio_indices

        # Start generation in separate thread
        generation_start_time = time.time()
        generation_thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        generation_thread.start()
        
        # Process streaming tokens
        generated_text = ""
        audio_chunks = []
        chunk_id = 0
        current_audio_tokens = []
        
        first_token_time = None
        first_text_token_time = None
        first_audio_token_time = None
        
        log_event("STREAM_INFER", task_type, "🎯 Starting real-time token processing...")
        
        for new_text in streamer:
            current_time = time.time()
            token_latency = current_time - generation_start_time
            
            # Track first token
            if first_token_time is None:
                first_token_time = token_latency
                log_event("STREAM_TIMING", task_type, f"🎯 FIRST TOKEN at {first_token_time:.3f}s")
            
            generated_text += new_text
            
            # Check for audio tokens in the new text
            if "<|audio_" in new_text and first_audio_token_time is None:
                first_audio_token_time = token_latency
                log_event("STREAM_TIMING", task_type, f"🎵 FIRST AUDIO TOKEN at {first_audio_token_time:.3f}s")
            
            # Check for text tokens (non-audio content)
            clean_new_text = re.sub(r"<\|[^|]*\|>", "", new_text).strip()
            if clean_new_text and first_text_token_time is None:
                first_text_token_time = token_latency
                log_event("STREAM_TIMING", task_type, f"📝 FIRST TEXT TOKEN at {first_text_token_time:.3f}s: '{clean_new_text[:20]}'")
            
            # Process audio tokens for streaming
            audio_tokens_in_text = re.findall(r"<\|audio_(\d+)\|>", new_text)
            for token_str in audio_tokens_in_text:
                if token_str.strip():
                    try:
                        current_audio_tokens.append(int(token_str))
                    except ValueError:
                        log_event("STREAM_INFER", task_type, f"Warning: Invalid audio token '{token_str}'")
                        continue
                
                # Process audio chunks in real-time (every N tokens)
                if len(current_audio_tokens) >= stream_stride and self.streaming_processor:
                    chunk_audio = self.streaming_processor.process_audio_chunk(
                        current_audio_tokens.copy(), chunk_id
                    )
                    if chunk_audio is not None:
                        audio_chunks.append(chunk_audio)
                    chunk_id += 1
                    current_audio_tokens = []  # Reset for next chunk
            
            # Log progress every few tokens
            if len(generated_text) % 100 == 0:
                log_event("STREAM_PROGRESS", task_type, f"Generated {len(generated_text)} chars in {token_latency:.3f}s")

        # Wait for generation to complete
        generation_thread.join()
        generation_end_time = time.time()
        generation_time = generation_end_time - generation_start_time
        
        # Process any remaining audio tokens
        if current_audio_tokens and self.streaming_processor:
            chunk_audio = self.streaming_processor.process_audio_chunk(current_audio_tokens, chunk_id)
            if chunk_audio is not None:
                audio_chunks.append(chunk_audio)

        # Stop streaming processor
        if self.streaming_processor:
            self.streaming_processor.stop_streaming()

        # Combine all audio chunks
        final_audio = None
        if audio_chunks:
            final_audio = np.concatenate(audio_chunks)
            log_event("STREAM_INFER", task_type, f"🎵 Combined {len(audio_chunks)} audio chunks into {final_audio.shape[0]/sample_rate:.2f}s audio")

        # Calculate final metrics
        request_end_time = time.time()
        total_request_time = request_end_time - request_start_time
        
        # Log comprehensive timing results
        log_event("STREAM_TIMING_SUMMARY", task_type, "=== STREAMING TIMING RESULTS ===")
        log_event("STREAM_TIMING_SUMMARY", task_type, f"Mode: {'TURBO' if self.use_turbo else 'BOOST'}")
        log_event("STREAM_TIMING_SUMMARY", task_type, f"MTP config: {self.model.generation_config.mtp_inference_mode}")
        log_event("STREAM_TIMING_SUMMARY", task_type, f"Total request time: {total_request_time:.3f}s")
        log_event("STREAM_TIMING_SUMMARY", task_type, f"Generation time: {generation_time:.3f}s")
        
        if first_token_time:
            log_event("STREAM_TIMING_SUMMARY", task_type, f"🎯 First token latency: {first_token_time:.3f}s")
        if first_text_token_time:
            log_event("STREAM_TIMING_SUMMARY", task_type, f"📝 First text token latency: {first_text_token_time:.3f}s")
        if first_audio_token_time:
            log_event("STREAM_TIMING_SUMMARY", task_type, f"🎵 First audio token latency: {first_audio_token_time:.3f}s")
            
        if audio_chunks:
            log_event("STREAM_TIMING_SUMMARY", task_type, f"🎵 Audio chunks generated: {len(audio_chunks)}")
            log_event("STREAM_TIMING_SUMMARY", task_type, f"🎵 Total audio duration: {final_audio.shape[0]/sample_rate:.2f}s")
            
        log_event("STREAM_TIMING_SUMMARY", task_type, "=== END STREAMING TIMING ===")
        
        return generated_text, final_audio

def _launch_demo(s2s_engine):
    def predict_chatbot(chatbot, task_history, task, use_turbo_mode):
        if not task_history:
            return chatbot, task_history, None
            
        # Switch engine mode based on user selection
        s2s_engine.switch_mode(use_turbo_mode)
        mode_name = "TURBO" if use_turbo_mode else "BOOST"
        log_event("CHAT", task, f"Using {mode_name} mode")
            
        chat_query = task_history[-1][0]
        log_event("CHAT", task, f"Processing query: {chat_query}")

        try:
            # Detect input type
            audio_path = None
            message = ""
            
            if isinstance(chat_query, str) and is_wav(chat_query):
                audio_path = chat_query
                message = ""
                log_event("CHAT", task, f"Audio input detected: {audio_path}")
            elif isinstance(chat_query, (tuple, list)) and len(chat_query) > 0:
                if is_wav(chat_query[0]):
                    audio_path = chat_query[0]
                    message = ""
                    log_event("CHAT", task, f"Audio input detected (from tuple): {audio_path}")
                else:
                    audio_path = None
                    message = str(chat_query[0])
                    log_event("CHAT", task, f"Text input detected (from tuple): {message}")
            else:
                audio_path = None
                message = str(chat_query)
                log_event("CHAT", task, f"Text input detected: {message}")

            # Validate inputs
            if task == "ASR" and audio_path is None:
                response_text = "❌ ASR task requires audio input. Please upload or record an audio file."
                audio_file_path = None
            elif task == "TTS" and not message.strip():
                response_text = "❌ TTS task requires text input. Please enter some text to convert to speech."
                audio_file_path = None
            else:
                # Run streaming inference
                output, tts_speech = s2s_engine.run_infer_streaming(
                    audio_path=audio_path,
                    message=message,
                    task_type=task,
                    max_returned_tokens=2048
                )

                # Clean text display
                clean_text, num_segments, total_tokens = clean_text_display(output, task_type=task)

                # Format response
                if task == "TTS":
                    if tts_speech is not None:
                        response_text = f"✅ Text converted to speech successfully! ({mode_name} mode)"
                        if total_tokens > 0:
                            response_text += f"\n🎵 Generated {total_tokens} audio tokens in {num_segments} segments"
                    else:
                        response_text = "❌ Failed to generate speech audio"
                elif task == "ASR":
                    if clean_text:
                        response_text = f"📝 Transcription ({mode_name} mode):\n{clean_text}"
                    else:
                        response_text = "❌ Failed to transcribe audio"
                else:  # Spoken QA
                    if clean_text:
                        response_text = f"🤖 Response ({mode_name} mode):\n{clean_text}"
                        if total_tokens > 0:
                            response_text += f"\n🎵 Generated {total_tokens} audio tokens in {num_segments} segments"
                    else:
                        response_text = "❌ Failed to generate response"

                # Save audio file
                audio_file_path = None
                if tts_speech is not None:
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        audio_filename = f"output_{task.lower()}_{timestamp}.wav"
                        audio_file_path = os.path.join("outputs", audio_filename)
                        os.makedirs("outputs", exist_ok=True)
                        
                        import soundfile as sf
                        sf.write(audio_file_path, tts_speech, 16000)
                        log_event("CHAT", task, f"Audio saved to: {audio_file_path}")
                    except Exception as e:
                        log_event("CHAT", task, f"Error saving audio: {e}")
                        audio_file_path = None

        except Exception as e:
            log_event("CHAT", task, f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            response_text = f"❌ Error: {str(e)}"
            audio_file_path = None

        # Update chat history - Keep tuple format for Gradio compatibility
        task_history[-1] = (task_history[-1][0], response_text)
        chatbot.append((parse_text(str(task_history[-1][0])), parse_text(response_text)))
        
        return chatbot, task_history, audio_file_path

    # Create Gradio interface
    with gr.Blocks(title="VITA-Audio Optimized Demo") as demo:
        gr.Markdown("# 🚀 VITA-Audio Optimized Demo with Turbo Mode")
        gr.Markdown("**Features:** Real-time streaming, Turbo/Boost mode switching, comprehensive timing metrics")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Mode selection
                turbo_mode = gr.Checkbox(
                    label="🚀 Turbo Mode", 
                    value=True,
                    info="Turbo: ~295 tok/s, 3.4ms/token | Boost: ~170 tok/s, 5.9ms/token"
                )
                
                # Task selection
                task_selector = gr.Radio(
                    ["Spoken QA", "ASR", "TTS"],
                    label="Task Type",
                    value="Spoken QA",
                    info="Spoken QA: Audio→Text+Audio | ASR: Audio→Text | TTS: Text→Audio"
                )
                
            with gr.Column(scale=2):
                # Chat interface - Use default (tuple) format
                chatbot = gr.Chatbot(
                    label="Chat History", 
                    height=400
                )
                
        with gr.Row():
            with gr.Column():
                # Input components
                audio_input = gr.Audio(
                    label="🎤 Audio Input (for Spoken QA & ASR)",
                    type="filepath"
                )
                text_input = gr.Textbox(
                    label="💬 Text Input (for Spoken QA & TTS)",
                    placeholder="Enter your message here..."
                )
                
            with gr.Column():
                # Output audio
                audio_output = gr.Audio(
                    label="🔊 Generated Audio Output",
                    type="filepath"
                )
                
        with gr.Row():
            submit_btn = gr.Button("🚀 Submit", variant="primary")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        # State management
        task_history = gr.State([])

        def user_input(audio_file, text_msg, history, task):
            if audio_file:
                user_message = audio_file
                display_message = f"🎤 Audio file: {os.path.basename(audio_file)}"
            elif text_msg.strip():
                user_message = text_msg.strip()
                display_message = text_msg.strip()
            else:
                return history, history, ""
                
            history = history + [[user_message, None]]
            return history, history, ""

        def clear_chat():
            return [], [], None

        # Event handlers
        submit_btn.click(
            user_input,
            [audio_input, text_input, task_history, task_selector],
            [chatbot, task_history, text_input],
            queue=False
        ).then(
            predict_chatbot,
            [chatbot, task_history, task_selector, turbo_mode],
            [chatbot, task_history, audio_output]
        )
        
        clear_btn.click(
            clear_chat,
            [],
            [chatbot, task_history, audio_output]
        )

    return demo

if __name__ == "__main__":
    # Initialize the optimized S2S engine
    log_event("MAIN", "SYSTEM", "🚀 Initializing VITA-Audio Optimized Engine...")
    
    s2s_engine = S2SInferenceOptimized(
        model_name_or_path=model_name_or_path,
        audio_tokenizer_path=audio_tokenizer_path,
        audio_tokenizer_type=audio_tokenizer_type,
        flow_path=flow_path,
        use_turbo=True  # Start with Turbo mode by default
    )
    
    log_event("MAIN", "SYSTEM", "✅ Engine initialized successfully!")
    
    # Launch demo
    demo = _launch_demo(s2s_engine)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
