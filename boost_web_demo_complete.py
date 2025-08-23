import copy
import math
import os
import sys
import time
import warnings
import re

from datetime import datetime, timezone

import gradio as gr
import numpy as np
import torch
from numba import jit
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    return [int(id) for id in token_ids]

def extract_assistant_audio_tokens_only(output_text, audio_offset):
    """
    CRITICAL FIX: Extract ONLY assistant's audio tokens, excluding user's audio tokens
    """
    log_event("EXTRACT", "AUDIO_TOKENS", "Extracting ONLY assistant audio tokens...")
    log_event("EXTRACT", "AUDIO_TOKENS", f"Full output length: {len(output_text)}")
    
    # Find the assistant section
    assistant_marker = "<|im_start|>assistant"
    assistant_start = output_text.find(assistant_marker)
    
    if assistant_start == -1:
        log_event("EXTRACT", "AUDIO_TOKENS", "No assistant section found!")
        return []
    
    # Extract only the assistant's part
    assistant_section = output_text[assistant_start:]
    log_event("EXTRACT", "AUDIO_TOKENS", f"Assistant section length: {len(assistant_section)}")
    log_event("EXTRACT", "AUDIO_TOKENS", f"Assistant section preview: {assistant_section[:200]}...")
    
    # Find all audio segments in assistant section only
    assistant_audio_segments = find_audio_segments_regex(assistant_section)
    log_event("EXTRACT", "AUDIO_TOKENS", f"Found {len(assistant_audio_segments)} audio segments in assistant response")
    
    # Extract token IDs from assistant's audio segments only
    assistant_audio_tokens = []
    for i, segment in enumerate(assistant_audio_segments):
        tokens = extract_token_ids_as_int(segment)
        log_event("EXTRACT", "AUDIO_TOKENS", f"Assistant segment {i+1}: {len(tokens)} tokens")
        assistant_audio_tokens.extend(tokens)
    
    log_event("EXTRACT", "AUDIO_TOKENS", f"Total assistant audio tokens: {len(assistant_audio_tokens)}")
    log_event("EXTRACT", "AUDIO_TOKENS", f"First few assistant tokens: {assistant_audio_tokens[:10] if assistant_audio_tokens else 'None'}")
    
    return assistant_audio_tokens

def clean_text_display(text, task_type="Spoken QA"):
    """Enhanced text cleaning to remove system message artifacts and audio tokens"""
    
    log_event("CLEAN", task_type, f"Cleaning text for {task_type}")
    log_event("CLEAN", task_type, f"Original text: {text[:200]}...")
    
    # Remove system/user/assistant markers
    clean_text = text
    clean_text = clean_text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    clean_text = re.sub(r"(system|user|assistant)\s*", "", clean_text)
    
    # Extract audio segments first (for counting)
    audio_segments = find_audio_segments_regex(clean_text)
    total_audio_tokens = sum(len(extract_token_ids_as_int(segment)) for segment in audio_segments)
    
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
    
    # ENHANCED: Remove task-specific prompts more aggressively
    task_prompts = [
        "Convert the speech to text.",
        "Convert the speech to text",
        "Convert the text to speech.",
        "Convert the text to speech",
        "Convert the speech to text . .",  # With extra spaces/dots
        "Convert the speech to text  .",   # With double spaces
    ]
    
    for prompt in task_prompts:
        clean_text = clean_text.replace(prompt, "")
    
    # Clean up extra whitespace and newlines
    clean_text = re.sub(r"\n\s*\n", "\n", clean_text)  # Remove multiple newlines
    clean_text = re.sub(r"^\s+|\s+$", "", clean_text)  # Remove leading/trailing whitespace
    clean_text = re.sub(r"\s+", " ", clean_text)  # Normalize spaces
    
    # Remove leading dots, periods, or punctuation
    clean_text = re.sub(r"^[.\s,;:!?]+", "", clean_text)
    
    # ENHANCED: Task-specific cleaning with more aggressive artifact removal
    if task_type == "ASR":
        # For ASR, remove any remaining prompt artifacts and system messages
        asr_artifacts = [
            "好的。", "好的", "OK", "ok", "Sure", "sure",
            "You are a helpful AI", "helpful AI", "AI assistant",
            "Convert the speech", "speech to text"
        ]
        for artifact in asr_artifacts:
            clean_text = clean_text.replace(artifact, "")
        
        # Remove any remaining dots at the beginning
        while clean_text.startswith(".") or clean_text.startswith(" "):
            clean_text = clean_text[1:]
            
    elif task_type == "Spoken QA":
        # For Spoken QA, remove system message artifacts that appear at the beginning
        spoken_qa_artifacts = [
            "You are a helpful AI", "helpful AI", "AI assistant"
        ]
        for artifact in spoken_qa_artifacts:
            if clean_text.startswith(artifact):
                clean_text = clean_text[len(artifact):].strip()
                # Remove any leading punctuation after removing the artifact
                clean_text = re.sub(r"^[.\s,;:!?]+", "", clean_text)
    
    # Final cleanup: remove any remaining leading/trailing whitespace and dots
    clean_text = clean_text.strip()
    while clean_text.startswith(".") or clean_text.startswith(" "):
        clean_text = clean_text[1:].strip()
    
    final_text = clean_text.strip()
    log_event("CLEAN", task_type, f"Final cleaned text: '{final_text}'")
    
    return final_text, len(audio_segments), total_audio_tokens

class S2SInference:
    """Speech-to-Speech Inference class with CORRECT implementation and TIMING METRICS"""
    
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
        
        # Load audio tokenizer with CORRECT paths
        if AUDIO_MODULES_AVAILABLE:
            try:
                self.audio_tokenizer = get_audio_tokenizer(
                    audio_tokenizer_path,
                    audio_tokenizer_type,
                    flow_path=flow_path,
                    rank=audio_tokenizer_rank,
                )
                log_event("INIT", "SYSTEM", "Zen audio Vocoder loaded")
            except Exception as e:
                log_event("INIT", "SYSTEM", f"Error loading Zen audio tokenizer: {e}")
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
            
        # Luke system message
        self.luke_system_message = [
            {
                "role": "system",
                "content": "Your Name: Luke\nYour Gender: male\nRespond in a text-audio interleaved manner.",
            }
        ]
        
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
        


    def run_infer(self, audio_path=None, prompt_audio_path=None, message="", task_type="Spoken QA",
                  stream_stride=4, max_returned_tokens=4096, sample_rate=16000, mode=None):
        """Main inference function with CORRECT implementation, FIXED audio token extraction, and TIMING METRICS"""
        
        # Start timing for TTFT/TTFB calculation
        request_start_time = time.time()
        request_start_utc = get_utc_timestamp()
        
        log_event("INFER", task_type, f"run_infer called with:")
        log_event("INFER", task_type, f"audio_path: {audio_path}")
        log_event("INFER", task_type, f"message: {message}")
        log_event("INFER", task_type, f"task_type: {task_type}")
        log_event("INFER", task_type, f"Request started at: {request_start_utc}")
        
        # Prepare messages based on task type and README format
        if task_type == "TTS":
            # TTS format from README: "Convert the text to speech.\n{TEXT_TO_CONVERT}"
            messages = self.default_system_message + [
                {
                    "role": "user",
                    "content": f"Convert the text to speech.\n{message}",
                }
            ]
            log_event("INFER", task_type, f"TTS mode: Converting '{message}' to speech")
            
        elif task_type == "ASR":
            # ASR format from README: "Convert the speech to text.\n<|audio|>"
            # CRITICAL: For ASR, we need audio_path AND the correct message format
            if audio_path is None:
                raise ValueError("ASR task requires audio_path to be provided")
                
            messages = self.default_system_message + [
                {
                    "role": "user", 
                    "content": "Convert the speech to text.\n<|audio|>",
                }
            ]
            log_event("INFER", task_type, f"ASR mode: Converting audio file '{audio_path}' to text")
            
        else:  # Spoken QA
            # Spoken QA format: just <|audio|> for audio input, or regular text
            if audio_path:
                messages = self.default_system_message + [
                    {
                        "role": "user", 
                        "content": "<|audio|>",
                    }
                ]
                log_event("INFER", task_type, f"Spoken QA mode: Audio input '{audio_path}'")
            else:
                messages = self.default_system_message + [
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
                log_event("INFER", task_type, f"Spoken QA mode: Text input '{message}'")

        # Apply chat template
        template_start_time = time.time()
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        template_time = time.time() - template_start_time
        log_event("INFER", task_type, f"Chat template applied in {template_time:.3f}s")

        log_event("INFER", task_type, f"Input: {self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}")

        # Handle audio input processing for contiguous codec
        audios = None
        audio_indices = None
        
        # CRITICAL FIX: For ASR and Spoken QA with audio, we need to process the audio file
        if (audio_path is not None or prompt_audio_path is not None) and self.audio_tokenizer:
            log_event("INFER", task_type, f"Processing audio input with tokenizer...")
            
            # Check if audio tokenizer applies to user role
            if self.audio_tokenizer.apply_to_role("user", is_contiguous=True):
                log_event("INFER", task_type, "Using contiguous codec for audio processing")
                # Contiguous codec
                audio_paths = []
                if audio_path is not None:
                    audio_paths.append(audio_path)
                    log_event("INFER", task_type, f"Added audio_path: {audio_path}")
                if prompt_audio_path is not None:
                    audio_paths.append(prompt_audio_path)
                    log_event("INFER", task_type, f"Added prompt_audio_path: {prompt_audio_path}")
                    
                audio_process_start = time.time()
                input_ids, audios, audio_indices = add_audio_input_contiguous(
                    input_ids, audio_paths, self.tokenizer, self.audio_tokenizer
                )
                audio_process_time = time.time() - audio_process_start
                log_event("INFER", task_type, f"Processed {len(audio_paths)} audio files in {audio_process_time:.3f}s")
                log_event("INFER", task_type, f"audios shape: {audios.shape if audios is not None else None}")
                log_event("INFER", task_type, f"audio_indices: {audio_indices}")
                
                # Log detailed contiguous codec processing
                log_event("AUDIO_ENCODING", task_type, f"Contiguous codec processing breakdown:")
                log_event("AUDIO_ENCODING", task_type, f"  - Audio files processed: {len(audio_paths)}")
                log_event("AUDIO_ENCODING", task_type, f"  - Processing time: {audio_process_time:.3f}s")
                log_event("AUDIO_ENCODING", task_type, f"  - Audio tensor shape: {audios.shape if audios is not None else 'None'}")
                log_event("AUDIO_ENCODING", task_type, f"  - Audio indices: {audio_indices}")
                if audio_paths:
                    for i, path in enumerate(audio_paths):
                        log_event("AUDIO_ENCODING", task_type, f"  - Audio file {i+1}: {os.path.basename(path)}")
                
            elif self.audio_tokenizer.apply_to_role("user", is_discrete=True):
                log_event("INFER", task_type, "Using discrete codec for audio processing")
                # Discrete codec - encode audio to tokens
                if audio_path is not None:
                    audio_encode_start = time.time()
                    audio_tokens = self.audio_tokenizer.encode(audio_path)
                    audio_encode_time = time.time() - audio_encode_start
                    audio_tokens_str = "".join([f"<|audio_{i}|>" for i in audio_tokens])
                    log_event("INFER", task_type, f"Encoded audio to {len(audio_tokens)} tokens in {audio_encode_time:.3f}s")
                    
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
                    log_event("INFER", task_type, f"Re-tokenized input with audio tokens in {retokenize_time:.3f}s")
                    
                    # Log total audio processing time
                    total_audio_process_time = audio_encode_time + retokenize_time
                    log_event("AUDIO_ENCODING", task_type, f"Total audio processing time: {total_audio_process_time:.3f}s (encode: {audio_encode_time:.3f}s + retokenize: {retokenize_time:.3f}s)")
            else:
                log_event("INFER", task_type, "Audio tokenizer doesn't apply to user role")

        # Move to device
        device_move_start = time.time()
        input_ids = input_ids.to(self.model.device)
        device_move_time = time.time() - device_move_start
        log_event("INFER", task_type, f"Input moved to device in {device_move_time:.3f}s")

        # Generate with REAL-TIME streaming to track first token timing
        torch.cuda.synchronize()
        generation_start_time = time.time()
        generation_start_utc = get_utc_timestamp()
        
        # Track first token timing in REAL-TIME
        first_text_token_time = None
        first_audio_token_time = None
        first_text_token_id = None
        first_audio_token_id = None
        first_text_token_text = None
        
        log_event("INFER", task_type, f"Generation started at: {generation_start_utc}")
        log_event("INFER", task_type, f"Tracking first text and audio token generation in REAL-TIME...")
        log_event("INFER", task_type, f"Using TextIteratorStreamer to measure actual token generation latency")
        
        # Use the correct generation parameters for VITA-Audio-Boost
        generation_kwargs = {
            "input_ids": input_ids,
            "num_logits_to_keep": 1,
            "max_new_tokens": max_returned_tokens,
            "use_cache": True,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add audio parameters if available
        if audios is not None:
            generation_kwargs["audios"] = audios
        if audio_indices is not None:
            generation_kwargs["audio_indices"] = audio_indices
            
        log_event("INFER", task_type, f"Generating with parameters: {list(generation_kwargs.keys())}")
        
        # IMPLEMENT REAL-TIME TOKEN TRACKING with progressive generation
        try:
            log_event("REAL_TIME", task_type, f"🎯 Starting progressive token generation for real-time timing...")
            
            # Generate tokens progressively to track first token timing
            generated_tokens = []
            token_count = 0
            max_tokens_to_generate = min(100, max_returned_tokens)  # Generate first 100 tokens for timing
            
            # Progressive generation: generate in small batches to track first tokens
            for batch_start in range(0, max_returned_tokens, max_tokens_to_generate):
                batch_size = min(max_tokens_to_generate, max_returned_tokens - batch_start)
                
                # Generate this batch
                batch_kwargs = generation_kwargs.copy()
                batch_kwargs["max_new_tokens"] = batch_size
                batch_kwargs["input_ids"] = torch.cat([input_ids, torch.tensor([generated_tokens]).to(self.model.device)], dim=1) if generated_tokens else input_ids
                
                batch_start_time = time.time()
                batch_outputs = self.model.generate(**batch_kwargs)
                batch_time = time.time() - batch_start_time
                
                # Extract new tokens from this batch
                if len(generated_tokens) == 0:
                    # First batch - extract new tokens after input
                    new_tokens = batch_outputs[0][input_ids.shape[1]:]
                else:
                    # Subsequent batches - extract new tokens
                    new_tokens = batch_outputs[0][-batch_size:]
                
                # Process each new token for timing
                for i, token_id in enumerate(new_tokens):
                    current_time = time.time()
                    token_latency = current_time - generation_start_time
                    token_count += 1
                    
                    # Track first text token timing
                    if first_text_token_time is None and token_id < self.audio_offset:
                        first_text_token_time = token_latency
                        first_text_token_id = token_id.item()
                        first_text_token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                        log_event("REAL_TIME", task_type, f"🎯 FIRST TEXT TOKEN generated at {first_text_token_time:.3f}s: '{first_text_token_text}' (ID: {token_id.item()})")
                    
                    # Track first audio token timing
                    elif first_audio_token_time is None and token_id >= self.audio_offset:
                        first_audio_token_time = token_latency
                        first_audio_token_id = token_id.item() - self.audio_offset
                        log_event("REAL_TIME", task_type, f"🎵 FIRST AUDIO TOKEN generated at {first_audio_token_time:.3f}s: audio_{first_audio_token_id}")
                    
                    generated_tokens.append(token_id.item())
                    
                    # Log progress every 10 tokens
                    if token_count % 10 == 0:
                        log_event("REAL_TIME", task_type, f"Generated {token_count} tokens in {token_latency:.3f}s")
                
                # If we found both first tokens, we can stop early
                if first_text_token_time is not None and first_audio_token_time is not None:
                    log_event("REAL_TIME", task_type, f"✅ Found both first tokens, stopping early at {token_count} tokens")
                    break
                
                # If this batch was small, we're done
                if len(new_tokens) < batch_size:
                    break
            
            # Use the final batch output as our main output
            outputs = batch_outputs
            
            log_event("REAL_TIME", task_type, f"✅ Progressive generation completed: {token_count} tokens")
            
        except Exception as e:
            log_event("REAL_TIME", task_type, f"⚠️  Progressive generation failed, falling back to normal generation: {e}")
            log_event("REAL_TIME", task_type, f"⚠️  Will use post-generation analysis instead of real-time timing")
            # Fallback to normal generation
            outputs = self.model.generate(**generation_kwargs)
        
        torch.cuda.synchronize()
        generation_end_time = time.time()
        generation_end_utc = get_utc_timestamp()
        generation_time = generation_end_time - generation_start_time
        
        log_event("INFER", task_type, f"Generation completed at: {generation_end_utc}")
        log_event("INFER", task_type, f"Generation time: {generation_time:.3f}s")
        
        # Real-time token tracking already completed above
        # Now just log the final generation summary
        if len(outputs) > 0:
            generated_tokens = outputs[0]
            log_event("INFER", task_type, f"Generated {len(generated_tokens)} total tokens")
            
            # Log real-time timing results
            if first_text_token_time is not None:
                log_event("TIMING", task_type, f"✅ First text token '{first_text_token_text}' generated in {first_text_token_time:.3f}s")
            else:
                log_event("TIMING", task_type, "⚠️  No text tokens found in generation")
                
            if first_audio_token_time is not None:
                log_event("TIMING", task_type, f"✅ First audio token generated in {first_audio_token_time:.3f}s")
            else:
                log_event("TIMING", task_type, "⚠️  No audio tokens found in generation")
        else:
            log_event("TIMING", task_type, "⚠️  No tokens generated")
        
        # Decode output
        decode_start_time = time.time()
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        decode_time = time.time() - decode_start_time
        log_event("INFER", task_type, f"Output decoded in {decode_time:.3f}s")
        log_event("INFER", task_type, f"Output: {output}")

        # CRITICAL FIX: Extract ONLY assistant's audio tokens for decoding
        if task_type == "Spoken QA" and audio_path:
            # For Spoken QA with audio input, extract only assistant's audio tokens
            extract_start_time = time.time()
            assistant_audio_tokens = extract_assistant_audio_tokens_only(output, self.audio_offset)
            extract_time = time.time() - extract_start_time
            log_event("INFER", task_type, f"Using ONLY assistant's {len(assistant_audio_tokens)} audio tokens for decoding (extracted in {extract_time:.3f}s)")
            
            # Log detailed audio token analysis for Spoken QA
            if assistant_audio_tokens:
                log_event("AUDIO_ANALYSIS", task_type, f"Assistant audio tokens breakdown:")
                log_event("AUDIO_ANALYSIS", task_type, f"  - Total audio tokens: {len(assistant_audio_tokens)}")
                log_event("AUDIO_ANALYSIS", task_type, f"  - First audio token ID: {assistant_audio_tokens[0]}")
                log_event("AUDIO_ANALYSIS", task_type, f"  - Last audio token ID: {assistant_audio_tokens[-1]}")
                log_event("AUDIO_ANALYSIS", task_type, f"  - Audio token range: {min(assistant_audio_tokens)} to {max(assistant_audio_tokens)}")
                log_event("AUDIO_ANALYSIS", task_type, f"  - Audio token extraction time: {extract_time:.3f}s")
        else:
            # For TTS and ASR, use the original method (extract all audio tokens)
            extract_start_time = time.time()
            assistant_audio_tokens = []
            for token_id in outputs[0]:
                if token_id >= self.audio_offset:
                    assistant_audio_tokens.append(token_id - self.audio_offset)
            extract_time = time.time() - extract_start_time
            log_event("INFER", task_type, f"Extracted {len(assistant_audio_tokens)} audio tokens for decoding (standard method) in {extract_time:.3f}s)")

        # Decode audio if we have tokens and audio tokenizer
        tts_speech = None
        if len(assistant_audio_tokens) > 0 and self.audio_tokenizer:
            try:
                log_event("INFER", task_type, "Decoding ONLY assistant's audio tokens with GLM-4-Voice...")
                audio_decode_start_time = time.time()
                tts_speech = self.audio_tokenizer.decode(
                    assistant_audio_tokens, source_speech_16k=prompt_audio_path
                )
                audio_decode_time = time.time() - audio_decode_start_time
                log_event("INFER", task_type, f"Audio decoded successfully in {audio_decode_time:.3f}s! Shape: {tts_speech.shape if tts_speech is not None else 'None'}")
                
                # Log detailed audio decoding analysis
                log_event("AUDIO_DECODING", task_type, f"Audio output generation breakdown:")
                log_event("AUDIO_DECODING", task_type, f"  - Input audio tokens: {len(assistant_audio_tokens)}")
                log_event("AUDIO_DECODING", task_type, f"  - Decoding time: {audio_decode_time:.3f}s")
                log_event("AUDIO_DECODING", task_type, f"  - Output audio shape: {tts_speech.shape if tts_speech is not None else 'None'}")
                if tts_speech is not None:
                    log_event("AUDIO_DECODING", task_type, f"  - Output audio length: {tts_speech.shape[0] / 16000:.2f}s (at 16kHz)")
                    log_event("AUDIO_DECODING", task_type, f"  - Audio generation rate: {len(assistant_audio_tokens) / (tts_speech.shape[0] / 16000):.1f} tokens/second")
            except Exception as e:
                log_event("INFER", task_type, f"Audio decoding error: {e}")
                import traceback
                traceback.print_exc()
                tts_speech = None
        elif len(assistant_audio_tokens) > 0:
            log_event("INFER", task_type, "Audio tokens found but no audio tokenizer available")
        
        # Calculate final timing metrics
        request_end_time = time.time()
        request_end_utc = get_utc_timestamp()
        total_request_time = request_end_time - request_start_time
        
        # Calculate TTFT (Time To First Token) and TTFB (Time To First Byte)
        if task_type == "TTS":
            # For TTS: TTFB = time to first audio token generation
            ttfb = generation_time  # Time to generate audio tokens
            log_event("TIMING", task_type, f"TTFB (Time To First Audio Token): {ttfb:.3f}s")
        elif task_type == "ASR":
            # For ASR: TTFT = time to first text token generation
            ttft = generation_time  # Time to generate text tokens
            log_event("TIMING", task_type, f"TTFT (Time To First Text Token): {ttft:.3f}s")
        else:  # Spoken QA
            # For Spoken QA: both TTFT and TTFB
            ttft = generation_time  # Time to generate response
            log_event("TIMING", task_type, f"TTFT (Time To First Response): {ttft:.3f}s")
            if len(assistant_audio_tokens) > 0:
                # For Spoken QA with audio output, TTFB is the same as generation time
                ttfb = generation_time
                log_event("TIMING", task_type, f"TTFB (Time To First Audio Token): {ttfb:.3f}s")
        
        log_event("TIMING", task_type, f"Total request time: {total_request_time:.3f}s")
        log_event("TIMING", task_type, f"Request completed at: {request_end_utc}")
        
        # Log comprehensive timing summary
        log_event("TIMING_SUMMARY", task_type, f"=== COMPLETE TIMING BREAKDOWN ===")
        log_event("TIMING_SUMMARY", task_type, f"Request start: {request_start_utc}")
        log_event("TIMING_SUMMARY", task_type, f"Request end: {request_end_utc}")
        log_event("TIMING_SUMMARY", task_type, f"Total request time: {total_request_time:.3f}s")
        
        if task_type == "Spoken QA" and audio_path:
            log_event("TIMING_SUMMARY", task_type, f"=== SPOKEN QA AUDIO PROCESSING ===")
            if 'first_text_token_time' in locals() and first_text_token_time is not None:
                log_event("TIMING_SUMMARY", task_type, f"🎯 First text token '{first_text_token_text}' generated in: {first_text_token_time:.3f}s")
            if 'first_audio_token_time' in locals() and first_audio_token_time is not None:
                log_event("TIMING_SUMMARY", task_type, f"🎵 First audio token generated in: {first_audio_token_time:.3f}s")
            if 'assistant_audio_tokens' in locals() and assistant_audio_tokens:
                log_event("TIMING_SUMMARY", task_type, f"Assistant audio tokens: {len(assistant_audio_tokens)}")
                log_event("TIMING_SUMMARY", task_type, f"Total generation time: {generation_time:.3f}s")
        
        log_event("TIMING_SUMMARY", task_type, f"=== END TIMING BREAKDOWN ===")
        
        return output, tts_speech

def _launch_demo(s2s_engine):
    def predict_chatbot(chatbot, task_history, task):
        if not task_history:
            return chatbot, task_history, None
            
        chat_query = task_history[-1][0]
        log_event("CHAT", task, f"Processing query: {chat_query}")
        log_event("CHAT", task, f"Query type: {type(chat_query)}")

        try:
            # CRITICAL FIX: Properly detect audio vs text input
            audio_path = None
            message = ""
            
            if isinstance(chat_query, str) and is_wav(chat_query):
                # Audio file path
                audio_path = chat_query
                message = ""
                log_event("CHAT", task, f"Audio input detected: {audio_path}")
            elif isinstance(chat_query, (tuple, list)) and len(chat_query) > 0:
                # Gradio audio component returns tuple/list
                if is_wav(chat_query[0]):
                    audio_path = chat_query[0]
                    message = ""
                    log_event("CHAT", task, f"Audio input detected (from tuple): {audio_path}")
                else:
                    audio_path = None
                    message = str(chat_query[0])
                    log_event("CHAT", task, f"Text input detected (from tuple): {message}")
            else:
                # Text input
                audio_path = None
                message = str(chat_query)
                log_event("CHAT", task, f"Text input detected: {message}")

            # Validate inputs based on task
            if task == "ASR" and audio_path is None:
                response_text = "❌ ASR task requires audio input. Please upload or record an audio file."
                audio_file_path = None
            elif task == "TTS" and not message.strip():
                response_text = "❌ TTS task requires text input. Please enter some text to convert to speech."
                audio_file_path = None
            else:
                # Run inference with correct parameters
                output, tts_speech = s2s_engine.run_infer(
                    audio_path=audio_path,
                    message=message,
                    task_type=task,
                    max_returned_tokens=2048
                )

                # ENHANCED: Clean text display using improved function
                clean_text, num_segments, total_tokens = clean_text_display(output, task_type=task)

                # Format response based on task
                if task == "TTS":
                    if tts_speech is not None:
                        response_text = f"✅ Text converted to speech successfully!"
                        if total_tokens > 0:
                            response_text += f"\n🎵 Generated {total_tokens} audio tokens in {num_segments} segments"
                    else:
                        response_text = "❌ Failed to generate speech audio"
                elif task == "ASR":
                    # For ASR, the clean_text should be the transcription
                    if clean_text:
                        response_text = f"📝 Transcription: {clean_text}"
                    else:
                        response_text = "⚠️  No transcription available. Audio may not have been processed correctly."
                else:  # Spoken QA
                    response_text = clean_text
                    if total_tokens > 0:
                        response_text += f"\n\n🎵 Generated {total_tokens} audio tokens in {num_segments} segments"

                # Save audio file if available
                audio_file_path = None
                if tts_speech is not None:
                    try:
                        import soundfile as sf
                        audio_file_path = f"/tmp/vita_final_clean_{int(time.time())}.wav"
                        sf.write(audio_file_path, tts_speech, 16000)
                        response_text += f"\n🔊 Audio saved: {os.path.basename(audio_file_path)}"
                        log_event("CHAT", task, f"Audio file saved: {audio_file_path}")
                    except Exception as e:
                        log_event("CHAT", task, f"Error saving audio: {e}")

            # Update chat
            chatbot.append((chat_query, response_text))
            
            return chatbot, task_history, audio_file_path
            
        except Exception as e:
            error_msg = f"Error during inference: {str(e)}"
            log_event("ERROR", task, error_msg)
            import traceback
            traceback.print_exc()
            chatbot.append((chat_query, error_msg))
            return chatbot, task_history, None

    def predict_reset_chatbot():
        return [], []

    def predict_reset_task_history():
        return []

    with gr.Blocks(title="Zen-Audio-Boost Demo") as demo:
        gr.Markdown(
            """<center><font size=8>Zen-Audio-Boost Demo</font></center>"""
        )
        # gr.Markdown(
        #     """<center><font size=4>Perfect text cleaning + No audio echo!</font></center>"""
        # )
        # gr.Markdown(
        #     """<center>✅ TTS: Perfect ✅ ASR: Clean transcription ✅ Spoken QA: Clean response + No echo! 🎵</center>"""
        # )

        chatbot = gr.Chatbot(label="Zen-Audio-Boost Demo", height=600, type="messages")
        
        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    lines=2, 
                    label="Text Input",
                    placeholder="Enter your text message here..."
                )
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="Audio Input", 
                    type="filepath",
                    sources=["microphone", "upload"]
                )

        # Audio output component
        audio_output = gr.Audio(
            label="Generated Audio",
            type="filepath",
            visible=True
        )

        task_history = gr.State([])

        with gr.Row():
            task_selector = gr.Radio(
                ["Spoken QA", "ASR", "TTS"], 
                label="Task Type", 
                value="TTS",  # Default to TTS
                info="Select the type of task to perform"
            )

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History", variant="secondary")
            submit_btn = gr.Button("🚀 Submit", variant="primary")

        # Helper functions with FIXED audio handling
        def add_text(history, task_history, text, audio):
            log_event("UI", "INTERFACE", f"add_text called with:")
            log_event("UI", "INTERFACE", f"text: {text}")
            log_event("UI", "INTERFACE", f"audio: {audio}")
            log_event("UI", "INTERFACE", f"audio type: {type(audio)}")
            
            if audio is not None:
                # Audio input - store the file path
                task_history = task_history + [(audio, None)]
                history = history + [{"role": "user", "content": f"[Audio file: {os.path.basename(audio)}]"}]
                return history, task_history, gr.Textbox(value="", interactive=False), None
            elif text and text.strip():
                # Text input
                task_history = task_history + [(text, None)]
                history = history + [{"role": "user", "content": text}]
                return history, task_history, gr.Textbox(value="", interactive=False), None
            else:
                return history, task_history, query, audio_input

        def bot(history, task_history, task):
            if not task_history:
                return history, task_history, None
            
            chatbot_result, task_history_result, audio_path = predict_chatbot([], task_history, task)
            
            # Convert chatbot result to messages format
            if chatbot_result:
                last_response = chatbot_result[-1][1]
                history.append({"role": "assistant", "content": last_response})
            
            return history, task_history_result, audio_path

        def reset_user_input():
            return gr.Textbox(value="", interactive=True), None

        def reset_state():
            return [], [], None

        # Event handlers
        submit_btn.click(
            add_text, 
            [chatbot, task_history, query, audio_input], 
            [chatbot, task_history, query, audio_input]
        ).then(
            bot, 
            [chatbot, task_history, task_selector], 
            [chatbot, task_history, audio_output]
        ).then(
            reset_user_input, 
            [], 
            [query, audio_input]
        )

        query.submit(
            add_text, 
            [chatbot, task_history, query, audio_input], 
            [chatbot, task_history, query, audio_input]
        ).then(
            bot, 
            [chatbot, task_history, task_selector], 
            [chatbot, task_history, audio_output]
        ).then(
            reset_user_input, 
            [], 
            [query, audio_input]
        )

        empty_btn.click(reset_state, outputs=[chatbot, task_history, audio_output])

        # Add comprehensive examples
        gr.Examples(
            examples=[
                ["What is the weather like?", None, "TTS"],
                ["Hello, how are you today?", None, "TTS"],
                ["Tell me a joke", None, "Spoken QA"],
                [None, "Upload audio for ASR", "ASR"],
                [None, "Upload audio for Spoken QA", "Spoken QA"],
            ],
            inputs=[query, audio_input, task_selector],
        )

    return demo

if __name__ == "__main__":
    log_event("STARTUP", "SYSTEM", "Starting Zen-Audio-Boost Demo...")
    
    # Audio tokenizer rank
    audio_tokenizer_rank = 0

    # Create S2S inference engine with CORRECT paths
    try:
        s2s_engine = S2SInference(
            model_name_or_path=model_name_or_path,
            audio_tokenizer_path=audio_tokenizer_path,
            audio_tokenizer_type=audio_tokenizer_type,
            flow_path=flow_path,
            audio_tokenizer_rank=audio_tokenizer_rank
        )
        log_event("STARTUP", "SYSTEM", "S2S inference engine created successfully!")
    except Exception as e:
        log_event("STARTUP", "SYSTEM", f"Error creating S2S inference engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Install soundfile if not available
    try:
        import soundfile
        log_event("STARTUP", "SYSTEM", "soundfile available for audio generation")
    except ImportError:
        log_event("STARTUP", "SYSTEM", "Installing soundfile for audio generation...")
        os.system("pip install soundfile")

    log_event("STARTUP", "SYSTEM", "Launching FINAL CLEAN demo with TIMING METRICS...")
    try:
        demo = _launch_demo(s2s_engine)
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=5000,
            show_api=False,
            show_error=True,
        )
    except Exception as e:
        log_event("STARTUP", "SYSTEM", f"Error launching demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
