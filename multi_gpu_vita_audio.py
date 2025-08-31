#!/usr/bin/env python3
"""
Multi-GPU Optimized VITA-Audio Implementation (FIXED)
- GPU 0: Text generation (main model)
- GPU 1: Audio decoding (GLM4Voice tokenizer)
- GPU 2: Audio processing pipeline (optional)
- GPU 3: Reserved for future optimizations

Expected Performance Improvement:
- Audio chunk time: 922ms → ~200ms (4-5x faster)
- Parallel processing of text and audio
- Better GPU utilization
"""

import os
import sys
import time
import torch
import torch.multiprocessing as mp
import queue
import threading
from datetime import datetime, timezone
import gradio as gr
import numpy as np
from typing import Optional, List, Dict, Any
import logging
import re
import json

# Add VITA-Audio to path
sys.path.append(".")
sys.path.append("vita_audio")

# Add GLM-4-Voice paths
sys.path.append("third_party/GLM-4-Voice/")
sys.path.append("third_party/GLM-4-Voice/cosyvoice/")
sys.path.append("third_party/GLM-4-Voice/third_party/Matcha-TTS/")

# Import VITA-Audio modules (using correct imports from working code)
from vita_audio.data.processor.audio_processor import add_audio_input_contiguous
from vita_audio.tokenizer import get_audio_tokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_event(category: str, task_type: str, message: str):
    """Log events with UTC timestamp"""
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"{timestamp} [{category}] {task_type}: {message}")

class MultiGPUAudioProcessor:
    """Audio processor that runs on dedicated GPU"""
    
    def __init__(self, audio_tokenizer, device="cuda:1"):
        self.audio_tokenizer = audio_tokenizer
        self.device = device
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        
        # Move audio tokenizer to dedicated GPU
        if hasattr(self.audio_tokenizer, 'to'):
            self.audio_tokenizer = self.audio_tokenizer.to(device)
        
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
        """Background worker for audio processing"""
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
                    
                    # Move tokens to audio processing GPU
                    if isinstance(audio_tokens, list):
                        audio_tokens = torch.tensor(audio_tokens, device=self.device)
                    else:
                        audio_tokens = audio_tokens.to(self.device)
                    
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
                    'tokens': len(audio_tokens)
                }
                
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                log_event("MULTI_GPU", "AUDIO", f"Audio worker error: {e}")
                continue
    
    def process_audio_chunk_async(self, chunk_id: int, audio_tokens: List[int], start_time: float):
        """Submit audio chunk for async processing"""
        job = (chunk_id, audio_tokens, start_time)
        self.audio_queue.put(job)
    
    def get_completed_chunk(self, timeout=0.1):
        """Get completed audio chunk if available"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class MultiGPUVITAAudio:
    """Multi-GPU VITA-Audio implementation"""
    
    def __init__(self, 
                 model_path: str = "./models/VITA-MLLM/VITA-Audio-Boost",
                 audio_tokenizer_path: str = "./models/THUDM/glm-4-voice-tokenizer",
                 flow_path: str = "./models/THUDM/glm-4-voice-decoder",
                 text_gpu: str = "cuda:0",
                 audio_gpu: str = "cuda:1",
                 use_turbo: bool = True):
        
        self.model_path = model_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.flow_path = flow_path
        self.text_gpu = text_gpu
        self.audio_gpu = audio_gpu
        self.use_turbo = use_turbo
        
        # Initialize components
        self._load_text_model()
        self._load_audio_components()
        self._setup_multi_gpu_processing()
        
        log_event("MULTI_GPU", "INIT", f"Multi-GPU VITA-Audio initialized")
        log_event("MULTI_GPU", "INIT", f"Text model on {text_gpu}, Audio processing on {audio_gpu}")
    
    def _load_text_model(self):
        """Load text generation model on primary GPU"""
        log_event("MULTI_GPU", "INIT", f"Loading text model on {self.text_gpu}...")
        
        # Set primary GPU
        torch.cuda.set_device(self.text_gpu)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # Load model on primary GPU (using AutoModelForCausalLM like working code)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map={
                "": self.text_gpu  # Force all layers to primary GPU
            },
            trust_remote_code=True
        )
        
        # Configure MTP mode
        if self.use_turbo:
            self.model.generation_config.mtp_inference_mode = [1, 10]  # Turbo
            log_event("MULTI_GPU", "INIT", "🚀 TURBO MODE enabled: [1, 10]")
        else:
            self.model.generation_config.mtp_inference_mode = [1, 10, 4, 10]  # Boost
            log_event("MULTI_GPU", "INIT", "⚡ BOOST MODE enabled: [1, 10, 4, 10]")
        
        log_event("MULTI_GPU", "INIT", f"Text model loaded on {self.text_gpu}")
    
    def _load_audio_components(self):
        """Load audio components on dedicated GPU"""
        log_event("MULTI_GPU", "INIT", f"Loading audio components on {self.audio_gpu}...")
        
        # Set audio GPU
        torch.cuda.set_device(self.audio_gpu)
        
        # Load audio tokenizer using the working code's approach
        self.audio_tokenizer = get_audio_tokenizer(
            audio_tokenizer_type="glm4voice",
            audio_tokenizer_path=self.audio_tokenizer_path,
            flow_path=self.flow_path,
            device=self.audio_gpu
        )
        
        log_event("MULTI_GPU", "INIT", f"Audio components loaded on {self.audio_gpu}")
    
    def _setup_multi_gpu_processing(self):
        """Setup multi-GPU processing pipeline"""
        # Initialize multi-GPU audio processor
        self.audio_processor = MultiGPUAudioProcessor(
            self.audio_tokenizer, 
            device=self.audio_gpu
        )
        
        # Start background audio processing
        self.audio_processor.start_worker()
        
        # Audio processing state
        self.pending_chunks = {}
        self.completed_chunks = {}
        self.chunk_counter = 0
        
        log_event("MULTI_GPU", "INIT", "Multi-GPU processing pipeline initialized")
    
    def warmup_audio_decoder(self):
        """Warmup audio decoder on dedicated GPU"""
        log_event("MULTI_GPU", "WARMUP", "🔥 Starting multi-GPU audio decoder warmup...")
        
        warmup_start = time.time()
        
        # Find warmup audio files
        warmup_files = self._find_warmup_audio_files()
        
        if warmup_files:
            for i, warmup_file in enumerate(warmup_files[:2]):  # Use 2 files
                try:
                    log_event("MULTI_GPU", "WARMUP", f"Warming up with: {os.path.basename(warmup_file)}")
                    
                    # Encode audio to tokens (on audio GPU)
                    with torch.cuda.device(self.audio_gpu):
                        encode_start = time.time()
                        warmup_tokens = self.audio_tokenizer.encode(warmup_file)
                        encode_time = time.time() - encode_start
                        
                        # Decode tokens to audio (on audio GPU)
                        decode_start = time.time()
                        warmup_audio = self.audio_tokenizer.decode(warmup_tokens[:4])
                        decode_time = time.time() - decode_start
                    
                    log_event("MULTI_GPU", "WARMUP", 
                             f"Warmup {i+1}: encode={encode_time:.3f}s, decode={decode_time:.3f}s, tokens=4")
                    
                except Exception as e:
                    log_event("MULTI_GPU", "WARMUP", f"Warmup file {warmup_file} failed: {e}")
        else:
            # Fallback: warmup with dummy tokens
            log_event("MULTI_GPU", "WARMUP", "No warmup files found, using dummy tokens")
            with torch.cuda.device(self.audio_gpu):
                dummy_tokens = torch.randint(0, 16384, (4,), device=self.audio_gpu)
                decode_start = time.time()
                self.audio_tokenizer.decode(dummy_tokens)
                decode_time = time.time() - decode_start
                log_event("MULTI_GPU", "WARMUP", f"Dummy warmup: decode={decode_time:.3f}s")
        
        warmup_total = time.time() - warmup_start
        log_event("MULTI_GPU", "WARMUP", f"🔥 Multi-GPU audio decoder warmup completed in {warmup_total:.3f}s")
    
    def _find_warmup_audio_files(self):
        """Find available audio files for warmup"""
        warmup_files = []
        asset_dir = "asset"
        
        if os.path.exists(asset_dir):
            for file in os.listdir(asset_dir):
                if file.endswith(('.wav', '.mp3')):
                    file_path = os.path.join(asset_dir, file)
                    if os.path.getsize(file_path) > 1000:  # Skip tiny files
                        warmup_files.append(file_path)
        
        return warmup_files[:3]  # Limit to 3 files
    
    def run_infer_streaming(self, audio_path: str, task_type: str = "Spoken QA"):
        """Run streaming inference with multi-GPU optimization"""
        log_event("MULTI_GPU", task_type, "🎬 Starting multi-GPU streaming inference")
        log_event("MULTI_GPU", task_type, f"Text GPU: {self.text_gpu}, Audio GPU: {self.audio_gpu}")
        
        start_time = time.time()
        
        # Reset state
        self.pending_chunks = {}
        self.completed_chunks = {}
        self.chunk_counter = 0
        
        try:
            # Apply chat template (on CPU first)
            chat_template = self.tokenizer.apply_chat_template([
                {"role": "system", "content": "Your Name: Luke\nYour Gender: male\n\nRespond in a text-audio interleaved manner."},
                {"role": "user", "content": ""}
            ], add_generation_prompt=True, tokenize=False)
            
            # Process audio input on audio GPU
            with torch.cuda.device(self.audio_gpu):
                audio_start = time.time()
                input_ids, audios, audio_indices = add_audio_input_contiguous(
                    [chat_template],
                    [audio_path],
                    self.tokenizer,
                    self.audio_tokenizer
                )
                audio_time = time.time() - audio_start
                log_event("MULTI_GPU", task_type, f"Audio processing completed in {audio_time:.3f}s on {self.audio_gpu}")
            
            # Move input to text GPU for generation
            with torch.cuda.device(self.text_gpu):
                input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.text_gpu)
                
                # Setup streaming
                streamer = TextIteratorStreamer(
                    self.tokenizer, 
                    skip_prompt=True, 
                    skip_special_tokens=False
                )
                
                # Generation parameters
                generation_kwargs = {
                    "input_ids": input_ids,
                    "streamer": streamer,
                    "max_new_tokens": 8192,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                
                # Start generation in background thread
                generation_thread = threading.Thread(
                    target=self.model.generate, 
                    kwargs=generation_kwargs,
                    daemon=True
                )
                generation_thread.start()
                
                # Process streaming tokens
                generated_text = ""
                current_audio_tokens = []
                first_token_time = None
                first_text_token_time = None
                first_audio_token_time = None
                stream_stride = 4  # Process audio every 4 tokens
                
                log_event("MULTI_GPU", task_type, "🎯 Starting real-time token processing...")
                
                for new_text in streamer:
                    current_time = time.time()
                    
                    # Track first token
                    if first_token_time is None:
                        first_token_time = current_time - start_time
                        log_event("MULTI_GPU", task_type, f"🎯 FIRST TOKEN at {first_token_time:.3f}s")
                    
                    # Track first text token
                    if new_text.strip() and first_text_token_time is None:
                        first_text_token_time = current_time - start_time
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
                                    first_audio_token_time = current_time - start_time
                                    log_event("MULTI_GPU", task_type, f"🎵 FIRST AUDIO TOKEN at {first_audio_token_time:.3f}s")
                                
                            except ValueError:
                                continue
                    
                    # Process audio chunks when we have enough tokens
                    if len(current_audio_tokens) >= stream_stride:
                        chunk_tokens = current_audio_tokens[:stream_stride]
                        current_audio_tokens = current_audio_tokens[stream_stride:]
                        
                        # Submit for async audio processing
                        self.audio_processor.process_audio_chunk_async(
                            self.chunk_counter, 
                            chunk_tokens, 
                            start_time
                        )
                        
                        self.pending_chunks[self.chunk_counter] = {
                            'tokens': chunk_tokens,
                            'submit_time': current_time
                        }
                        
                        self.chunk_counter += 1
                    
                    # Check for completed audio chunks
                    self._process_completed_chunks(task_type)
                    
                    generated_text += new_text
                
                # Process remaining audio tokens
                if current_audio_tokens:
                    self.audio_processor.process_audio_chunk_async(
                        self.chunk_counter, 
                        current_audio_tokens, 
                        start_time
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
                
                total_time = time.time() - start_time
                
                # Log performance summary
                log_event("MULTI_GPU", task_type, "=== MULTI-GPU TIMING RESULTS ===")
                log_event("MULTI_GPU", task_type, f"Text GPU: {self.text_gpu}")
                log_event("MULTI_GPU", task_type, f"Audio GPU: {self.audio_gpu}")
                log_event("MULTI_GPU", task_type, f"Total request time: {total_time:.3f}s")
                log_event("MULTI_GPU", task_type, f"🎯 First token latency: {first_token_time:.3f}s")
                if first_text_token_time:
                    log_event("MULTI_GPU", task_type, f"📝 First text token latency: {first_text_token_time:.3f}s")
                if first_audio_token_time:
                    log_event("MULTI_GPU", task_type, f"🎵 First audio token latency: {first_audio_token_time:.3f}s")
                log_event("MULTI_GPU", task_type, f"🎵 Audio chunks generated: {len(self.completed_chunks)}")
                log_event("MULTI_GPU", task_type, "=== END MULTI-GPU TIMING ===")
                
                return self._clean_text_for_task(generated_text, task_type)
                
        except Exception as e:
            log_event("MULTI_GPU", task_type, f"Multi-GPU inference error: {e}")
            return f"Error: {e}"
    
    def _process_completed_chunks(self, task_type: str):
        """Process completed audio chunks from background worker"""
        while True:
            result = self.audio_processor.get_completed_chunk()
            if result is None:
                break
            
            chunk_id = result['chunk_id']
            chunk_time = result['chunk_time']
            total_time = result['total_time']
            tokens = result['tokens']
            
            # Calculate audio duration (assuming 50 tokens/second)
            audio_duration = tokens * 0.02  # Rough estimate
            
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
            self._process_completed_chunks(task_type)
            time.sleep(0.01)  # Small delay
        
        if self.pending_chunks:
            log_event("MULTI_GPU", task_type, f"Warning: {len(self.pending_chunks)} chunks still pending after timeout")
    
    def _clean_text_for_task(self, text: str, task_type: str) -> str:
        """Clean generated text for specific task"""
        # Remove audio tokens and system tokens
        text = re.sub(r'<\|audio_\d+\|>', '', text)
        text = re.sub(r'<\|begin_of_audio\|>', '', text)
        text = re.sub(r'<\|end_of_audio\|>', '', text)
        text = re.sub(r'<\|im_start\|>', '', text)
        text = re.sub(r'<\|im_end\|>', '', text)
        
        # Clean whitespace
        text = ' '.join(text.split())
        text = text.strip()
        
        log_event("MULTI_GPU", task_type, f"Final cleaned text: '{text}'")
        return text
    
    def switch_mode(self, use_turbo: bool):
        """Switch between Turbo and Boost modes"""
        self.use_turbo = use_turbo
        if use_turbo:
            self.model.generation_config.mtp_inference_mode = [1, 10]
            log_event("MULTI_GPU", "CONFIG", "🚀 Switched to TURBO MODE: [1, 10]")
        else:
            self.model.generation_config.mtp_inference_mode = [1, 10, 4, 10]
            log_event("MULTI_GPU", "CONFIG", "⚡ Switched to BOOST MODE: [1, 10, 4, 10]")
    
    def __del__(self):
        """Cleanup multi-GPU resources"""
        if hasattr(self, 'audio_processor'):
            self.audio_processor.stop_worker()

def create_gradio_interface():
    """Create Gradio interface for multi-GPU VITA-Audio"""
    
    # Initialize multi-GPU engine
    log_event("MULTI_GPU", "MAIN", "🔥 Initializing Multi-GPU VITA-Audio Engine...")
    
    engine = MultiGPUVITAAudio(
        use_turbo=True  # Start with Turbo mode
    )
    
    # Warmup audio decoder
    engine.warmup_audio_decoder()
    
    log_event("MULTI_GPU", "MAIN", "✅ Multi-GPU Engine initialized successfully!")
    
    def chat_interface(audio_input, task_selector, use_turbo_mode, history):
        """Main chat interface function"""
        if audio_input is None:
            return history, ""
        
        # Switch mode if needed
        engine.switch_mode(use_turbo_mode)
        
        mode_name = "TURBO" if use_turbo_mode else "BOOST"
        log_event("MULTI_GPU", task_selector, f"Using {mode_name} mode (🔥 Multi-GPU)")
        
        # Process audio input
        log_event("MULTI_GPU", task_selector, f"Processing query: {audio_input}")
        
        try:
            # Run multi-GPU inference
            response = engine.run_infer_streaming(audio_input, task_selector)
            
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
        gr.Markdown("**Optimized for 4x H100 80GB GPUs - Text on GPU:0, Audio on GPU:1**")
        
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
                    label="🚀 Turbo Mode (faster, [1,10] vs [1,10,4,10])",
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
        - **GPU 0**: Text generation (VITA-Audio model)
        - **GPU 1**: Audio decoding (GLM4Voice tokenizer)  
        - **Expected**: 4-5x faster audio chunk generation (~200ms vs 900ms)
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
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
