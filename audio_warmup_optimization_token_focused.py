#!/usr/bin/env python3
"""
VITA-Audio Token-Level Analysis (Simplified)
Focus on real-time token generation and sentence detection

Features:
- Real-time token display (text vs audio tokens)
- Accurate sentence detection on clean text only
- Detailed token-level logging
- Gradio UI showing token stream
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

# Sentence-ending punctuation for first sentence detection
SENTENCE_ENDINGS = ".!?:;"

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

def detect_sentence_completion(accumulated_clean_text):
    """
    Detect if accumulated clean text contains a complete sentence
    
    Args:
        accumulated_clean_text (str): Clean text accumulated so far
    
    Returns:
        tuple: (is_complete, sentence_text, remaining_text)
    """
    if len(accumulated_clean_text.strip()) < 3:
        return False, "", accumulated_clean_text
    
    # Look for sentence endings
    for i, char in enumerate(accumulated_clean_text):
        if char in SENTENCE_ENDINGS:
            # Found a sentence ending
            sentence = accumulated_clean_text[:i+1].strip()
            remaining = accumulated_clean_text[i+1:].strip()
            
            # Validate it's a meaningful sentence (at least 2 words)
            sentence_words = re.findall(r'\b\w+\b', sentence)
            if len(sentence_words) >= 2:
                return True, sentence, remaining
    
    return False, "", accumulated_clean_text

class TokenStreamAnalyzer:
    """Analyzes token stream in real-time"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset analyzer for new request"""
        self.accumulated_text = ""
        self.accumulated_clean_text = ""
        self.text_tokens = []
        self.audio_tokens = []
        self.token_timestamps = []
        
        # Timing tracking
        self.first_token_time = None
        self.first_text_token_time = None
        self.first_audio_token_time = None
        self.first_sentence_time = None
        self.first_sentence_text = ""
        self.sentence_detected = False
        
        self.generation_start_time = None
    
    def set_generation_start_time(self, start_time):
        """Set the generation start time"""
        self.generation_start_time = start_time
    
    def process_token_chunk(self, new_text, task_type="Spoken QA"):
        """Process a new token chunk from the stream"""
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
        
        # Track first text token
        if clean_text_chunk.strip() and self.first_text_token_time is None:
            self.first_text_token_time = token_latency
            log_event("TOKEN", task_type, f"📝 FIRST TEXT TOKEN at {self.first_text_token_time:.3f}s: '{clean_text_chunk.strip()}'")
        
        # Track first audio token
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
                    'time': current_time
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
            'first_sentence': self.first_sentence_text if self.sentence_detected else None
        }
    
    def get_summary(self, task_type="Spoken QA"):
        """Get timing summary"""
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
        
        log_event("SUMMARY", task_type, f"📊 Total text tokens: {len(self.text_tokens)}")
        log_event("SUMMARY", task_type, f"📊 Total audio tokens: {len(self.audio_tokens)}")
        log_event("SUMMARY", task_type, f"📊 Clean text length: {len(self.accumulated_clean_text)} chars")
        log_event("SUMMARY", task_type, "=== END SUMMARY ===")

class S2SInferenceTokenFocused:
    """Simplified VITA-Audio inference focused on token analysis"""
    
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
        
        log_event("INIT", "MAIN", f"Token-focused VITA-Audio initialized")
    
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
    
    def run_token_analysis(self, audio_path=None, message="", task_type="Spoken QA", max_returned_tokens=4096):
        """
        Run inference with detailed token-level analysis
        """
        request_start_time = time.time()
        log_event("REQUEST", task_type, f"🚀 Starting token-focused inference...")
        
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

        # Handle audio input processing
        audios = None
        audio_indices = None
        
        if audio_path is not None and self.audio_tokenizer:
            log_event("AUDIO_INPUT", task_type, f"Processing audio input...")
            
            if self.audio_tokenizer.apply_to_role("user", is_discrete=True):
                try:
                    audio_tokens = self.audio_tokenizer.encode(audio_path)
                    audio_tokens_str = "".join([f"<|audio_{i}|>" for i in audio_tokens])
                    log_event("AUDIO_INPUT", task_type, f"Encoded {len(audio_tokens)} audio tokens")
                    
                    # Replace <|audio|> with actual tokens
                    input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    input_text = input_text.replace("<|audio|>", f"<|begin_of_audio|>{audio_tokens_str}<|end_of_audio|>")
                    input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
                    
                except Exception as e:
                    log_event("AUDIO_INPUT", task_type, f"Audio processing error: {e}")

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
        
        log_event("GENERATION", task_type, "🎯 Starting token stream processing...")
        
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
        
        # Get final summary
        self.token_analyzer.get_summary(task_type)
        
        total_time = time.time() - request_start_time
        log_event("REQUEST", task_type, f"Total request time: {total_time:.3f}s")
        
        return {
            'final_text': self.token_analyzer.accumulated_clean_text,
            'stream_results': stream_results,
            'summary': {
                'first_token_time': self.token_analyzer.first_token_time,
                'first_text_token_time': self.token_analyzer.first_text_token_time,
                'first_audio_token_time': self.token_analyzer.first_audio_token_time,
                'first_sentence_time': self.token_analyzer.first_sentence_time,
                'first_sentence_text': self.token_analyzer.first_sentence_text,
                'total_text_tokens': len(self.token_analyzer.text_tokens),
                'total_audio_tokens': len(self.token_analyzer.audio_tokens),
                'total_time': total_time
            }
        }

def create_token_analysis_interface():
    """Create Gradio interface for token analysis"""
    
    # Initialize engine
    log_event("MAIN", "INIT", "🔥 Initializing Token-Focused VITA-Audio...")
    
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
        """Analyze token generation"""
        if audio_input is None and not text_input.strip():
            return "Please provide audio input or text input", "", "", "", ""
        
        try:
            # Run analysis
            if audio_input:
                result = s2s_engine.run_token_analysis(
                    audio_path=audio_input,
                    task_type=task_selector
                )
            else:
                result = s2s_engine.run_token_analysis(
                    message=text_input,
                    task_type=task_selector
                )
            
            # Format results for display
            final_text = result['final_text']
            summary = result['summary']
            
            # Create token stream display
            token_stream = ""
            text_tokens_display = ""
            audio_tokens_display = ""
            
            for item in result['stream_results']:
                new_text = item['new_text']
                timestamp = item['timestamp']
                
                # Extract components
                clean_text = extract_clean_text_only(new_text)
                audio_tokens = extract_audio_tokens_from_chunk(new_text)
                
                if clean_text.strip():
                    text_tokens_display += f"[{timestamp:.3f}s] {clean_text}\n"
                
                if audio_tokens:
                    audio_tokens_display += f"[{timestamp:.3f}s] {audio_tokens}\n"
                
                token_stream += f"[{timestamp:.3f}s] {new_text}\n"
            
            # Create timing summary
            first_text_time = f"{summary['first_text_token_time']:.3f}s" if summary['first_text_token_time'] else 'N/A'
            first_audio_time = f"{summary['first_audio_token_time']:.3f}s" if summary['first_audio_token_time'] else 'N/A'
            first_sentence_time = f"{summary['first_sentence_time']:.3f}s" if summary['first_sentence_time'] else 'N/A'
            
            timing_summary = f"""=== TIMING SUMMARY ===
🎯 First Token: {summary['first_token_time']:.3f}s
📝 First Text Token: {first_text_time}
🎵 First Audio Token: {first_audio_time}
🎉 First Sentence: {first_sentence_time}

📊 COUNTS:
Text Tokens: {summary['total_text_tokens']}
Audio Tokens: {summary['total_audio_tokens']}
Total Time: {summary['total_time']:.3f}s

🎉 FIRST SENTENCE:
{summary['first_sentence_text'] or 'No complete sentence detected'}"""
            
            return final_text, text_tokens_display, audio_tokens_display, token_stream, timing_summary
            
        except Exception as e:
            error_msg = f"Error: {e}"
            log_event("ERROR", task_selector, error_msg)
            return error_msg, "", "", "", ""
    
    # Create interface
    with gr.Blocks(title="Zen-Audio Token Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Zen-Audio Token-Level Analysis")
        gr.Markdown("**Real-time token stream analysis**")
        
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
                
                analyze_btn = gr.Button("Analyze Tokens", variant="primary")
            
            with gr.Column(scale=2):
                # Outputs
                final_text = gr.Textbox(
                    label="Final Clean Text",
                    lines=3
                )
                
                timing_summary = gr.Textbox(
                    label="Timing Summary",
                    lines=10
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
        
        # Event handler
        analyze_btn.click(
            analyze_tokens,
            inputs=[audio_input, task_selector, text_input],
            outputs=[final_text, text_tokens, audio_tokens, token_stream, timing_summary]
        )
    
    return demo

if __name__ == "__main__":
    log_event("MAIN", "START", "🚀 Starting Token-Focused VITA-Audio Analysis...")
    
    # Create and launch interface
    demo = create_token_analysis_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True
    )
