#!/usr/bin/env python3
"""
Zen-Audio Enhanced Full Demo with VITA-Audio-Boost
This version properly implements:
1. Detailed system message like audio_warmup_optimization_with_zen_decoder_with_sys.py
2. Uses text tokens for ElevenLabs (Zen Decoder) - NOT audio tokens
3. Shows detailed logs like the original zen decoder implementation
4. Properly handles GLM-4 decoder for audio tokens vs ElevenLabs for text
5. Different behavior for ASR vs Spoken QA tasks

Key Technical Details:
- VITA-Audio-Boost generates both text tokens and audio tokens
- Text tokens are used for ElevenLabs TTS (Zen Decoder)
- Audio tokens should be decoded using GLM-4 decoder (but we bypass this for ElevenLabs)
- ASR task: Audio input -> Text output (no audio generation)
- Spoken QA task: Audio/Text input -> Text + Audio output
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
import uuid
from datetime import datetime, timezone
from threading import Thread
from queue import Queue
import threading
import queue
import json
from typing import List, Dict, Any, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from numba import jit
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Zen Decoder integration (ElevenLabs TTS)
try:
    from elevenlabs.client import ElevenLabs
    ZEN_DECODER_AVAILABLE = True
except ImportError:
    print("⚠️  Zen Decoder not available - install elevenlabs package")
    ZEN_DECODER_AVAILABLE = False

# Add GLM-4-Voice paths (for proper audio token decoding)
if True:
    sys.path.append("third_party/GLM-4-Voice/")
    sys.path.append("third_party/GLM-4-Voice/cosyvoice/")
    sys.path.append("third_party/GLM-4-Voice/third_party/Matcha-TTS/")

    audio_tokenizer_path = "./models/THUDM/glm-4-voice-tokenizer"
    flow_path = "./models/THUDM/glm-4-voice-decoder"
    audio_tokenizer_type = "glm4voice"
    #model_name_or_path = "./models/VITA-MLLM/VITA-Audio-Boost"
    model_name_or_path = "/mnt/output/debt_collection_training/debt_collection_finetune_20250915_180254/checkpoint-1000"

# Import VITA-Audio modules
try:
    from vita_audio.data.processor.audio_processor import add_audio_input_contiguous
    from vita_audio.tokenizer import get_audio_tokenizer
    AUDIO_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Audio modules not available: {e}")
    AUDIO_MODULES_AVAILABLE = False

# Detailed system prompt (matching audio_warmup_optimization_with_zen_decoder_with_sys.py)
DETAILED_SYSTEM_PROMPT = """
You're Susan, an AI assistant for Stratforge. Your primary task is to interact with the customer as a support agent, and gather information on customer query, grievances, issues etc. You dont have access to any system or database to check, so you can only take information but cant check it in system. Be empathic in responses as customer might be frustrated. 
 
[Context]
You're engaged with the customer who has called you with his problem with EV charging station. Stay focused on this context and provide relevant information. Once connected to a customer, proceed to the Conversation Flow section. Do not invent information not drawn from the context. Answer only questions related to the context. Be empathic in responses as customer might be frustrated. 

Greeting Message : Hello, Thank you for calling Startforge. I am Susan. How can i help you today? 

After the Customer response to Greeting Message, Acknowledge the issue the customer is facing and ask "Can i have your Name and Callback number incase this call gets disconnected."

Once the customer provides his name and Number, Acknowledge his response by repeating the name and phone number for confirmation.

After the customer confirms the details are right, then move to the issue and its solution. You can only solve basic issues and whenever you feel that this requires further support transfer it to concerned department. 

Customer can have following types of issues:
1. Billing Issues
  - Customers inquiring about charges on their accounts, including unexpected fees.
  - Clarification requests on billing statements.
  - Complaints about overcharges or incorrect deductions.
  - Issues related to payment processing or failed transactions.
  - Subscription plan concerns (upgrading, downgrading, cancellation).
  - Charging session not recorded in the billing system.

2. Technical Issues
  - Charging station not working or malfunctioning.
  - Issues with the mobile app or website.
  - Problems with RFID cards or contactless payments.
  - Charging cable or connector problems.
  - Display screen issues on charging stations.
  - Network connectivity problems.

3. Account Issues
  - Problems logging into accounts.
  - Issues with account registration or verification.
  - Password reset requests.
  - Profile information updates.
  - Account suspension or deactivation concerns.

4. General Inquiries
  - Information about charging station locations.
  - Questions about charging speeds and compatibility.
  - Pricing and subscription plan details.
  - How to use the charging stations.
  - Availability and reservation queries.

Remember to be empathetic, professional, and helpful throughout the conversation. Always gather necessary information before attempting to provide solutions.
"""

# Sentence-ending punctuation for sentence detection
SENTENCE_ENDINGS = ".!?:;,"

def get_utc_timestamp():
    """Get current UTC timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()

def log_event(event_type, task_type, message, **kwargs):
    """Log event with UTC timestamp and task context (matching original zen decoder)"""
    timestamp = get_utc_timestamp()
    task_info = f"[{task_type}]" if task_type else ""
    print(f"{timestamp} {task_info} {event_type}: {message}")

def clean_text_display(text, task_type="Spoken QA"):
    """Clean text for display by removing artifacts and special tokens (enhanced version)"""
    if not text:
        return "", 0, 0
    
    # Remove chat template tags and system messages
    clean_text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text, flags=re.DOTALL)
    clean_text = re.sub(r'<\|user\|>.*?<\|assistant\|>', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'<\|system\|>.*?<\|assistant\|>', '', clean_text, flags=re.DOTALL)
    
    # Count and remove audio segments
    audio_segments = re.findall(r'<\|begin_of_audio\|>.*?<\|end_of_audio\|>', clean_text, flags=re.DOTALL)
    audio_tokens = re.findall(r'<\|audio_(\d+)\|>', text)
    total_audio_tokens = len(audio_tokens)
    
    # Remove all audio-related tags
    clean_text = re.sub(r'<\|begin_of_audio\|>.*?<\|end_of_audio\|>', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'<\|audio_\d+\|>', '', clean_text)
    clean_text = re.sub(r'<\|audio\|>', '', clean_text)
    
    # Remove system message artifacts
    clean_text = re.sub(r"You are a helpful AI assistant\.", "", clean_text)
    clean_text = re.sub(r"You're Susan.*?conversation\.", "", clean_text, flags=re.DOTALL)
    
    # Clean up whitespace and punctuation
    clean_text = re.sub(r"^\s+|\s+$", "", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)
    clean_text = re.sub(r"^[.\s,;:!?]+", "", clean_text)
    
    # Task-specific cleaning
    if task_type == "ASR":
        asr_artifacts = [
            "好的。", "好的", "OK", "ok", "Sure", "sure",
            "You are a helpful AI", "helpful AI", "AI assistant",
            "Convert the speech", "speech to text"
        ]
        for artifact in asr_artifacts:
            clean_text = clean_text.replace(artifact, "")
        
        while clean_text.startswith(".") or clean_text.startswith(" "):
            clean_text = clean_text[1:]
    
    # Final cleanup
    clean_text = clean_text.strip()
    
    return clean_text, len(audio_segments), total_audio_tokens

def extract_audio_tokens_from_chunk(text_chunk):
    """Extract audio tokens from a text chunk (matching original zen decoder)"""
    audio_tokens = re.findall(r'<\|audio_(\d+)\|>', text_chunk)
    return [int(token) for token in audio_tokens]

def detect_sentence_completion(accumulated_clean_text):
    """
    Detect if accumulated clean text contains a complete sentence
    (matching original zen decoder logic)
    """
    if len(accumulated_clean_text.strip()) < 1:
        return False, "", accumulated_clean_text
    
    # Look for sentence endings
    for i, char in enumerate(accumulated_clean_text):
        if char in SENTENCE_ENDINGS:
            # Found a sentence ending
            sentence = accumulated_clean_text[:i+1].strip()
            remaining = accumulated_clean_text[i+1:].strip()
            
            # Accept even single words with punctuation for immediate processing
            sentence_words = re.findall(r'\b\w+\b', sentence)
            if len(sentence_words) >= 1:
                return True, sentence, remaining
    
    return False, "", accumulated_clean_text

class EnhancedTokenAnalyzer:
    """
    Enhanced token analyzer that matches the original zen decoder implementation
    with detailed logging and proper audio/text token separation
    """
    
    def __init__(self, zen_decoder_api_key: str = None):
        self.zen_decoder = None
        self.server_start_time = time.time()
        self.generation_start_time = None
        self.audio_encoding_time = 0
        
        # Token tracking (matching original zen decoder)
        self.text_tokens = []
        self.audio_tokens = []
        self.accumulated_clean_text = ""
        self.first_token_time = None
        self.first_text_token_time = None
        self.first_audio_token_time = None
        self.first_sentence_time = None
        self.first_zen_chunk_time = None
        
        # Zen Decoder setup (ElevenLabs TTS)
        if ZEN_DECODER_AVAILABLE and zen_decoder_api_key:
            try:
                self.zen_decoder = ElevenLabs(api_key=zen_decoder_api_key)
                self.sentence_counter = 0
                self.audio_files = []
                log_event("ZEN_DECODER_INIT", "MAIN", "🎵 Zen Decoder (ElevenLabs TTS) initialized successfully")
            except Exception as e:
                log_event("ZEN_DECODER_ERROR", "MAIN", f"Failed to initialize Zen Decoder: {e}")
                self.zen_decoder = None
        else:
            log_event("ZEN_DECODER_INIT", "MAIN", "⚠️ Zen Decoder not available")
            self.zen_decoder = None
        
        # Create output directory
        os.makedirs("/tmp/zen_enhanced_generated", exist_ok=True)
    
    def set_generation_start_time(self, start_time):
        """Set the generation start time for timing measurements"""
        self.generation_start_time = start_time
    
    def set_audio_encoding_time(self, encoding_time):
        """Set the audio encoding time"""
        self.audio_encoding_time = encoding_time
    
    def process_token_chunk(self, new_text, task_type, current_time):
        """
        Process token chunk with detailed analysis (matching original zen decoder)
        
        Key Point: We use TEXT tokens for ElevenLabs TTS, NOT audio tokens
        Audio tokens from VITA-Audio-Boost should be decoded with GLM-4 decoder
        """
        cumulative_time = current_time - self.server_start_time
        
        # Set first token time
        if self.first_token_time is None:
            self.first_token_time = cumulative_time
            log_event("FIRST_TOKEN", task_type, f"🎯 First token at {cumulative_time:.3f}s from server start")
        
        # Extract and process audio tokens (but don't use them for TTS)
        audio_tokens_in_chunk = extract_audio_tokens_from_chunk(new_text)
        if audio_tokens_in_chunk:
            if self.first_audio_token_time is None:
                self.first_audio_token_time = cumulative_time
                log_event("FIRST_AUDIO_TOKEN", task_type, f"🎵 First audio token at {cumulative_time:.3f}s")
            
            for token in audio_tokens_in_chunk:
                self.audio_tokens.append({
                    'token': token,
                    'timestamp': cumulative_time,
                    'chunk_text': new_text
                })
            
            log_event("AUDIO_TOKENS", task_type, f"🎵 Found {len(audio_tokens_in_chunk)} audio tokens: {audio_tokens_in_chunk}")
        
        # Clean text and check for actual text content
        clean_chunk, _, _ = clean_text_display(new_text, task_type)
        
        if clean_chunk.strip():
            if self.first_text_token_time is None:
                self.first_text_token_time = cumulative_time
                log_event("FIRST_TEXT_TOKEN", task_type, f"📝 First text token at {cumulative_time:.3f}s")
            
            # Add to text tokens
            self.text_tokens.append({
                'text': clean_chunk,
                'timestamp': cumulative_time,
                'raw_chunk': new_text
            })
            
            # Update accumulated clean text
            self.accumulated_clean_text += clean_chunk
            
            # Check for sentence completion and generate audio using TEXT (not audio tokens)
            is_complete, sentence, remaining = detect_sentence_completion(self.accumulated_clean_text)
            
            if is_complete and sentence.strip():
                if self.first_sentence_time is None:
                    self.first_sentence_time = cumulative_time
                    log_event("FIRST_SENTENCE", task_type, f"🎉 First complete sentence at {cumulative_time:.3f}s: '{sentence}'")
                
                # Generate audio using Zen Decoder (ElevenLabs) with TEXT tokens
                if self.zen_decoder and task_type in ["Spoken QA", "TTS"]:
                    self._generate_zen_audio_chunk(sentence, task_type, cumulative_time)
                
                # Update accumulated text
                self.accumulated_clean_text = remaining
            
            log_event("TEXT_CHUNK", task_type, f"📝 Text chunk: '{clean_chunk}' (cumulative: {len(self.accumulated_clean_text)} chars)")
        
        return {
            'timestamp': cumulative_time,
            'clean_text': clean_chunk,
            'accumulated_text': self.accumulated_clean_text,
            'audio_tokens_count': len(audio_tokens_in_chunk),
            'text_tokens_count': len([t for t in self.text_tokens if t['text'].strip()]),
            'total_audio_tokens': len(self.audio_tokens)
        }
    
    def _generate_zen_audio_chunk(self, sentence, task_type, timestamp):
        """
        Generate audio chunk using Zen Decoder (ElevenLabs TTS) with TEXT tokens
        
        Important: This uses the cleaned TEXT, not the audio tokens from VITA-Audio-Boost
        """
        try:
            if self.first_zen_chunk_time is None:
                self.first_zen_chunk_time = timestamp
                log_event("FIRST_ZEN_CHUNK", task_type, f"🎵 First Zen Decoder chunk at {timestamp:.3f}s")
            
            self.sentence_counter += 1
            log_event("ZEN_DECODER", task_type, f"🎵 Generating audio for sentence {self.sentence_counter}: '{sentence[:50]}...'")
            
            # Generate audio using ElevenLabs TTS (Zen Decoder) with TEXT
            audio_response = self.zen_decoder.generate(
                text=sentence,
                voice="bRfSN6IjvoNM52ilGATs",  # Default voice
                model="eleven_monolingual_v1"
            )
            
            # Save audio file
            audio_file = f"/tmp/zen_enhanced_generated/sentence_{self.sentence_counter}_{int(timestamp*1000)}.wav"
            with open(audio_file, "wb") as f:
                for chunk in audio_response:
                    f.write(chunk)
            
            self.audio_files.append(audio_file)
            log_event("ZEN_DECODER", task_type, f"🎵 Audio saved: {audio_file}")
            
        except Exception as e:
            log_event("ZEN_DECODER_ERROR", task_type, f"Audio generation failed: {e}")
    
    def finalize_analysis(self):
        """Finalize analysis and process any remaining text"""
        if self.accumulated_clean_text.strip() and self.zen_decoder:
            log_event("FINALIZE", "ANALYSIS", f"Processing remaining text: '{self.accumulated_clean_text}'")
            self._generate_zen_audio_chunk(self.accumulated_clean_text.strip(), "FINALIZE", time.time() - self.server_start_time)
    
    def get_summary(self, task_type):
        """Get comprehensive summary (matching original zen decoder)"""
        return {
            'first_token_time': self.first_token_time,
            'first_text_token_time': self.first_text_token_time,
            'first_audio_token_time': self.first_audio_token_time,
            'first_sentence_time': self.first_sentence_time,
            'first_zen_chunk_time': self.first_zen_chunk_time,
            'audio_encoding_time': self.audio_encoding_time,
            'total_text_tokens': len([t for t in self.text_tokens if t['text'].strip()]),
            'total_audio_tokens': len(self.audio_tokens),
            'zen_sentences': self.sentence_counter,
            'total_time': time.time() - self.server_start_time,
            'first_sentence_text': self.get_first_sentence(),
            'audio_files': self.audio_files
        }
    
    def get_first_sentence(self):
        """Get the first complete sentence"""
        if not self.text_tokens:
            return None
        
        accumulated = ""
        for token in self.text_tokens:
            accumulated += token['text']
            is_complete, sentence, _ = detect_sentence_completion(accumulated)
            if is_complete:
                return sentence
        
        return None
    
    def get_latest_audio_file(self):
        """Get the latest generated audio file"""
        return self.audio_files[-1] if self.audio_files else None
    
    def get_complete_audio_file(self):
        """Get combined audio file (placeholder - would need audio processing)"""
        return self.get_latest_audio_file()  # For now, return latest

class EnhancedConversationHistory:
    """Enhanced conversation history with proper task handling"""
    
    def __init__(self, system_prompt: str = DETAILED_SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.task_history = []  # Matches original VITA-Audio format
        self.chatbot_history = []  # For display
        self.audio_files = []
        self.session_id = str(uuid.uuid4())
        self.creation_time = time.time()
        
        log_event("HISTORY", "INIT", f"Created enhanced conversation history: {self.session_id}")
    
    def add_user_turn(self, content: str, audio_path: Optional[str] = None):
        """Add user turn with proper formatting"""
        if audio_path:
            # Audio input
            task_entry = ((audio_path,), None)
            chatbot_entry = ((audio_path,), None)
            self.audio_files.append(audio_path)
            log_event("HISTORY", "USER_AUDIO", f"Added audio: {audio_path}")
        else:
            # Text input with punctuation handling
            task_text = content
            if len(content) >= 2 and content[-1] in SENTENCE_ENDINGS and content[-2] not in SENTENCE_ENDINGS:
                task_text = content[:-1]
            
            task_entry = (task_text, None)
            chatbot_entry = (content, None)
            log_event("HISTORY", "USER_TEXT", f"Added text: {content[:50]}...")
        
        self.task_history.append(task_entry)
        self.chatbot_history.append(chatbot_entry)
    
    def update_assistant_response(self, response: str):
        """Update assistant response"""
        if not self.task_history:
            return
        
        cleaned_response, _, _ = clean_text_display(response, "Spoken QA")
        
        # Update both histories
        last_task = self.task_history[-1]
        self.task_history[-1] = (last_task[0], cleaned_response)
        
        last_chatbot = self.chatbot_history[-1]
        self.chatbot_history[-1] = (last_chatbot[0], cleaned_response)
        
        log_event("HISTORY", "ASSISTANT", f"Updated response: {cleaned_response[:50]}...")
    
    def get_messages_for_model(self, task: str = "Spoken QA") -> Tuple[List[Dict[str, str]], List[str]]:
        """Get messages formatted for model input with task-specific handling"""
        messages = []
        audio_path_list = []
        
        # Add system message based on task (matching original VITA-Audio logic)
        if task == "Spoken QA":
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        # ASR and TTS don't use system messages in original VITA-Audio
        
        # Process task history
        for i, (q, a) in enumerate(self.task_history):
            if isinstance(q, (tuple, list)) and q[0] and q[0].lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
                # Audio input
                audio_path_list.append(q[0])
                if task == "Spoken QA":
                    messages.append({"role": "user", "content": "\n<|audio|>"})
                elif task == "TTS":
                    messages.append({"role": "user", "content": "\n<|audio|>"})
                elif task == "ASR":
                    messages.append({"role": "user", "content": "Convert the speech to text.\n<|audio|>"})
            else:
                # Text input
                if task == "Spoken QA":
                    messages.append({"role": "user", "content": q})
                elif task == "TTS":
                    messages.append({"role": "user", "content": f"Convert the text to speech.\n{q}"})
                elif task == "ASR":
                    messages.append({"role": "user", "content": f"{q}"})
            
            # Add assistant response if available
            if a is not None:
                messages.append({"role": "assistant", "content": a})
        
        return messages, audio_path_list
    
    def reset(self):
        """Reset conversation history"""
        self.task_history.clear()
        self.chatbot_history.clear()
        self.audio_files.clear()
        log_event("HISTORY", "RESET", f"Reset conversation history: {self.session_id}")
    
    def get_stats(self):
        """Get conversation statistics"""
        return {
            "session_id": self.session_id,
            "total_turns": len(self.task_history),
            "audio_turns": sum(1 for q, a in self.task_history if isinstance(q, (tuple, list))),
            "text_turns": sum(1 for q, a in self.task_history if not isinstance(q, (tuple, list))),
            "duration": time.time() - self.creation_time
        }

class EnhancedS2SEngine:
    """
    Enhanced S2S Engine with proper system message, detailed logging,
    and correct audio/text token handling
    """
    
    def __init__(self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, flow_path, zen_decoder_api_key=None):
        self.model_name_or_path = model_name_or_path
        self.conversation_history = EnhancedConversationHistory()
        self.server_start_time = time.time()
        
        # Load components
        self._load_model_components()
        
        # Initialize token analyzer with Zen Decoder
        self.token_analyzer = EnhancedTokenAnalyzer(zen_decoder_api_key)
        
        log_event("INIT", "ENGINE", "Enhanced S2S Engine initialized with detailed logging")
    
    def _load_model_components(self):
        """Load model components with detailed logging"""
        log_event("INIT", "TOKENIZER", "Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        
        log_event("INIT", "MODEL", "Loading VITA-Audio-Boost model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        
        # Load audio tokenizer for GLM-4 decoder (proper audio token processing)
        if AUDIO_MODULES_AVAILABLE:
            try:
                self.audio_tokenizer = get_audio_tokenizer(
                    audio_tokenizer_path,
                    audio_tokenizer_type,
                    flow_path=flow_path,
                    rank=0,
                )
                self.audio_tokenizer.load_model()
                log_event("INIT", "AUDIO", "GLM-4 Audio tokenizer loaded for proper audio token decoding")
            except Exception as e:
                log_event("INIT", "AUDIO_ERROR", f"Audio tokenizer failed: {e}")
                self.audio_tokenizer = None
        else:
            self.audio_tokenizer = None
        
        # Configure generation based on task
        self.model.generation_config.max_new_tokens = 8192
        self.model.generation_config.use_cache = True
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
    
    def process_conversation_turn(self, user_input: str = "", audio_path: Optional[str] = None, 
                                task: str = "Spoken QA", max_tokens: int = 2048):
        """
        Process conversation turn with enhanced logging and proper task handling
        
        Key behaviors by task:
        - ASR: Audio input -> Text output (no audio generation)
        - TTS: Text input -> Text + Audio output 
        - Spoken QA: Audio/Text input -> Text + Audio output (with conversation history)
        """
        
        start_time = time.time()
        log_event("CONVERSATION", "START", f"Processing {task} turn - Text: '{user_input[:50]}...', Audio: {audio_path is not None}")
        
        # Add user turn to history
        self.conversation_history.add_user_turn(user_input, audio_path)
        
        # Get messages and audio files
        messages, audio_path_list = self.conversation_history.get_messages_for_model(task)
        
        log_event("CONTEXT", "MESSAGES", f"Using {len(messages)} messages, {len(audio_path_list)} audio files for {task}")
        
        # Configure generation based on task (matching original VITA-Audio)
        if task == "Spoken QA":
            self.model.generation_config.do_sample = False
            log_event("CONFIG", task, "Configured for Spoken QA: do_sample=False")
        elif task == "TTS":
            self.model.generation_config.do_sample = True
            log_event("CONFIG", task, "Configured for TTS: do_sample=True")
        elif task == "ASR":
            self.model.generation_config.do_sample = False
            log_event("CONFIG", task, "Configured for ASR: do_sample=False")
        
        # Apply chat template
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            log_event("TEMPLATE", task, f"Applied chat template: {len(input_ids)} tokens")
        except Exception as e:
            log_event("TEMPLATE", "ERROR", f"Chat template error: {e}")
            return {"error": f"Template error: {e}"}
        
        # Process audio with detailed timing
        audios = None
        audio_indices = None
        audio_encoding_time = 0
        
        if audio_path_list and self.audio_tokenizer:
            log_event("AUDIO_ENCODING", task, f"🎧 Processing {len(audio_path_list)} audio files...")
            
            audio_encoding_start = time.time()
            try:
                input_ids, audios, audio_indices = add_audio_input_contiguous(
                    input_ids, audio_path_list, self.tokenizer, self.audio_tokenizer
                )
                audio_encoding_end = time.time()
                audio_encoding_time = audio_encoding_end - audio_encoding_start
                
                self.token_analyzer.set_audio_encoding_time(audio_encoding_time)
                
                cumulative_encoding_time = audio_encoding_end - self.server_start_time
                log_event("AUDIO_ENCODING", task, f"🎧 Audio encoded in {audio_encoding_time*1000:.1f}ms (cumulative: {cumulative_encoding_time:.3f}s)")
                
            except Exception as e:
                log_event("AUDIO_ENCODING", "ERROR", f"Audio processing failed: {e}")
                audio_encoding_time = 0
        
        # Convert to tensor and move to device
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.model.device)
        
        # Setup streaming
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=60.0,
            skip_prompt=True,
            skip_special_tokens=False
        )
        
        generation_kwargs = {
            "input_ids": input_ids,
            "audios": audios,
            "audio_indices": audio_indices,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "use_cache": True,
            "num_logits_to_keep": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        # Start generation
        generation_start_time = time.time()
        generation_start_cumulative = generation_start_time - self.server_start_time
        self.token_analyzer.set_generation_start_time(generation_start_time)
        
        log_event("GENERATION", task, f"🎯 Starting token stream processing at {generation_start_cumulative:.3f}s from server start...")
        
        generation_thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        generation_thread.start()
        
        # Process token stream with detailed analysis
        stream_results = []
        for new_text in streamer:
            current_time = time.time()
            result = self.token_analyzer.process_token_chunk(new_text, task, current_time)
            stream_results.append({
                'new_text': new_text,
                'timestamp': result['timestamp'],
                'result': result
            })
        
        # Wait for generation to complete
        generation_thread.join(timeout=10.0)
        
        # Finalize analysis
        self.token_analyzer.finalize_analysis()
        
        # Update conversation history
        self.conversation_history.update_assistant_response(self.token_analyzer.accumulated_clean_text)
        
        # Get comprehensive summary
        summary = self.token_analyzer.get_summary(task)
        
        total_time = time.time() - self.server_start_time
        log_event("REQUEST", task, f"Total request time: {total_time:.3f}s")
        
        return {
            'final_text': self.token_analyzer.accumulated_clean_text,
            'stream_results': stream_results,
            'zen_decoder_files': summary['audio_files'],
            'latest_audio_file': self.token_analyzer.get_latest_audio_file(),
            'complete_audio_file': self.token_analyzer.get_complete_audio_file(),
            'summary': summary,
            'conversation_history': self.conversation_history.chatbot_history,
            'conversation_stats': self.conversation_history.get_stats(),
            'task': task
        }
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history.reset()
        return "🔄 Conversation history reset successfully"

def create_enhanced_interface():
    """Create enhanced interface with proper system message and detailed logging"""
    
    log_event("MAIN", "INIT", "🔥 Initializing Enhanced Zen-Audio with detailed system prompt and logging...")
    
    # Replace with your actual Zen Decoder API key
    zen_decoder_api_key = "YOUR_ZEN_DECODER_API_KEY"
    
    engine = EnhancedS2SEngine(
        model_name_or_path=model_name_or_path,
        audio_tokenizer_path=audio_tokenizer_path,
        audio_tokenizer_type=audio_tokenizer_type,
        flow_path=flow_path,
        zen_decoder_api_key=zen_decoder_api_key
    )
    
    def process_enhanced_conversation(text_input, audio_input, task_selector, conversation_display):
        """Process conversation with enhanced logging and analysis"""
        try:
            if not text_input.strip() and audio_input is None:
                return (
                    conversation_display,
                    "Please provide either text or audio input.",
                    "",
                    "",
                    "",
                    None
                )
            
            # Process the conversation turn
            result = engine.process_conversation_turn(
                user_input=text_input,
                audio_path=audio_input,
                task=task_selector,
                max_tokens=2048
            )
            
            if 'error' in result:
                return (
                    conversation_display,
                    f"❌ Error: {result['error']}",
                    "",
                    "",
                    "",
                    None
                )
            
            # Update conversation display
            updated_conversation = result['conversation_history']
            
            # Create detailed displays (matching original zen decoder)
            final_text = result['final_text']
            
            # Text tokens display
            text_tokens_display = ""
            for token_info in engine.token_analyzer.text_tokens:
                text_tokens_display += f"[{token_info['timestamp']:.3f}s] {token_info['text']}\n"
            
            # Audio tokens display
            audio_tokens_display = ""
            for token_info in engine.token_analyzer.audio_tokens:
                audio_tokens_display += f"[{token_info['timestamp']:.3f}s] <|audio_{token_info['token']}|>\n"
            
            # Complete token stream (interleaved)
            all_tokens = []
            
            # Add text tokens
            for token_info in engine.token_analyzer.text_tokens:
                all_tokens.append({
                    'timestamp': token_info['timestamp'],
                    'content': token_info['text'],
                    'type': 'text'
                })
            
            # Add audio tokens
            for token_info in engine.token_analyzer.audio_tokens:
                all_tokens.append({
                    'timestamp': token_info['timestamp'],
                    'content': f"<|audio_{token_info['token']}|>",
                    'type': 'audio'
                })
            
            # Sort by timestamp and build display
            all_tokens.sort(key=lambda x: x['timestamp'])
            token_stream = ""
            for token in all_tokens:
                timestamp = token['timestamp']
                content = token['content']
                token_stream += f"[{timestamp:.3f}s] {content}\n"
            
            # Create comprehensive timing summary
            summary = result['summary']
            first_text_time = f"{summary['first_text_token_time']:.3f}s" if summary['first_text_token_time'] else 'N/A'
            first_audio_time = f"{summary['first_audio_token_time']:.3f}s" if summary['first_audio_token_time'] else 'N/A'
            first_sentence_time = f"{summary['first_sentence_time']:.3f}s" if summary['first_sentence_time'] else 'N/A'
            first_zen_chunk_time = f"{summary['first_zen_chunk_time']:.3f}s" if summary['first_zen_chunk_time'] else 'N/A'
            audio_encoding_time = f"{summary['audio_encoding_time']*1000:.1f}ms" if summary['audio_encoding_time'] > 0 else 'N/A'
            
            timing_summary = f"""=== ENHANCED TIMING SUMMARY ===
Task: {result['task']}
🎧 Audio Encoding Time: {audio_encoding_time}
🎯 First Token: {summary['first_token_time']:.3f}s
📝 First Text Token: {first_text_time}
🎵 First Audio Token: {first_audio_time}
🎉 First Sentence: {first_sentence_time}
🎵 First Zen Decoder Chunk: {first_zen_chunk_time}

📊 DETAILED COUNTS:
Text Tokens: {summary['total_text_tokens']}
Audio Tokens: {summary['total_audio_tokens']} (decoded by GLM-4, but TTS uses text)
Zen Decoder Sentences: {summary['zen_sentences']} (using TEXT tokens for ElevenLabs)
Total Time: {summary['total_time']:.3f}s

🎉 FIRST SENTENCE:
{summary['first_sentence_text'] or 'No complete sentence detected'}

📋 CONVERSATION STATS:
Session: {result['conversation_stats']['session_id'][:8]}...
Total Turns: {result['conversation_stats']['total_turns']}
Audio Turns: {result['conversation_stats']['audio_turns']}
Text Turns: {result['conversation_stats']['text_turns']}"""
            
            # Get audio output
            audio_output = result.get('complete_audio_file')
            
            return (
                updated_conversation,
                final_text,
                text_tokens_display,
                audio_tokens_display,
                token_stream,
                timing_summary,
                audio_output
            )
            
        except Exception as e:
            error_msg = f"Error: {e}"
            log_event("ERROR", task_selector, error_msg)
            return (
                conversation_display,
                error_msg,
                "",
                "",
                "",
                "",
                None
            )
    
    def reset_conversation():
        """Reset conversation"""
        message = engine.reset_conversation()
        return [], message, "", "", "", "", None
    
    # Create interface
    with gr.Blocks(title="Enhanced Zen-Audio Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Enhanced Zen-Audio with Detailed System Prompt & Logging")
        gr.Markdown("**Complete implementation matching audio_warmup_optimization_with_zen_decoder_with_sys.py with conversation history**")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Inputs
                audio_input = gr.Audio(
                    label="🎤 Audio Input", 
                    type="filepath",
                    format="wav"
                )
                
                text_input = gr.Textbox(
                    label="💭 Text Input",
                    placeholder="Type your message here...",
                    lines=2
                )
                
                task_selector = gr.Dropdown(
                    choices=["Spoken QA", "ASR", "TTS"],
                    value="Spoken QA",
                    label="Task Type"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Submit", variant="primary")
                    reset_btn = gr.Button("🔄 Reset Conversation", variant="secondary")
            
            with gr.Column(scale=2):
                # Conversation display
                conversation_display = gr.Chatbot(
                    label="💬 Conversation History",
                    height=300
                )
                
                # Final text output
                final_text = gr.Textbox(
                    label="📝 Final Clean Text",
                    lines=3
                )
        
        with gr.Row():
            with gr.Column():
                text_tokens = gr.Textbox(
                    label="📝 Text Tokens Stream (Used for ElevenLabs TTS)",
                    lines=8
                )
            
            with gr.Column():
                audio_tokens = gr.Textbox(
                    label="🎵 Audio Tokens Stream (From VITA-Audio-Boost, decoded by GLM-4)", 
                    lines=8
                )
        
        with gr.Row():
            token_stream = gr.Textbox(
                label="🔄 Complete Interleaved Token Stream",
                lines=10
            )
        
        with gr.Row():
            timing_summary = gr.Textbox(
                label="⏱️ Enhanced Timing Analysis",
                lines=15
            )
        
        # Audio output section
        with gr.Row():
            gr.Markdown("### 🎵 Zen Decoder Audio Output (Generated from TEXT tokens via ElevenLabs)")
            audio_output = gr.Audio(
                label="🔊 Generated Audio Response",
                type="filepath"
            )
        
        # Event handlers
        submit_btn.click(
            process_enhanced_conversation,
            inputs=[text_input, audio_input, task_selector, conversation_display],
            outputs=[conversation_display, final_text, text_tokens, audio_tokens, token_stream, timing_summary, audio_output]
        )
        
        text_input.submit(
            process_enhanced_conversation,
            inputs=[text_input, audio_input, task_selector, conversation_display],
            outputs=[conversation_display, final_text, text_tokens, audio_tokens, token_stream, timing_summary, audio_output]
        )
        
        reset_btn.click(
            reset_conversation,
            outputs=[conversation_display, final_text, text_tokens, audio_tokens, token_stream, timing_summary, audio_output]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Hello, I need help with my EV charging station.", None, "Spoken QA"],
                ["I'm having billing problems with my account.", None, "Spoken QA"],
                ["Convert this text to speech: Hello, how are you today?", None, "TTS"],
                [None, "example_audio.wav", "ASR"],
            ],
            inputs=[text_input, audio_input, task_selector]
        )
    
    return demo

if __name__ == "__main__":
    log_event("MAIN", "START", "🚀 Starting Enhanced Zen-Audio Analysis with detailed system prompt...")
    
    # Create and launch interface
    demo = create_enhanced_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=True,
        debug=True,
        show_error=True
    )
