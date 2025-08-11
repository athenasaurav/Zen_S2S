import copy
import math
import os
import sys
import time
import warnings
import re
from threading import Thread

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
    print(f"🎯 Extracting ONLY assistant audio tokens...")
    print(f"   - Full output length: {len(output_text)}")
    
    # Find the assistant section
    assistant_marker = "<|im_start|>assistant"
    assistant_start = output_text.find(assistant_marker)
    
    if assistant_start == -1:
        print("   - No assistant section found!")
        return []
    
    # Extract only the assistant's part
    assistant_section = output_text[assistant_start:]
    print(f"   - Assistant section length: {len(assistant_section)}")
    print(f"   - Assistant section preview: {assistant_section[:200]}...")
    
    # Find all audio segments in assistant section only
    assistant_audio_segments = find_audio_segments_regex(assistant_section)
    print(f"   - Found {len(assistant_audio_segments)} audio segments in assistant response")
    
    # Extract token IDs from assistant's audio segments only
    assistant_audio_tokens = []
    for i, segment in enumerate(assistant_audio_segments):
        tokens = extract_token_ids_as_int(segment)
        print(f"   - Assistant segment {i+1}: {len(tokens)} tokens")
        assistant_audio_tokens.extend(tokens)
    
    print(f"   - Total assistant audio tokens: {len(assistant_audio_tokens)}")
    print(f"   - First few assistant tokens: {assistant_audio_tokens[:10] if assistant_audio_tokens else 'None'}")
    
    return assistant_audio_tokens

def clean_text_display(text, task_type="Spoken QA"):
    """Enhanced text cleaning to remove system message artifacts and audio tokens"""
    
    print(f"🧹 Cleaning text for {task_type}")
    print(f"   - Original text: {text[:200]}...")
    
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
    print(f"   - Final cleaned text: '{final_text}'")
    
    return final_text, len(audio_segments), total_audio_tokens

class S2SInference:
    """Speech-to-Speech Inference class with CORRECT implementation"""
    
    def __init__(self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, flow_path, audio_tokenizer_rank=0):
        self.model_name_or_path = model_name_or_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.audio_tokenizer_type = audio_tokenizer_type
        self.flow_path = flow_path
        self.audio_tokenizer_rank = audio_tokenizer_rank
        
        # Load tokenizer
        print("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        print(f"✅ Tokenizer loaded: {self.tokenizer.__class__.__name__}")
        
        # Load model
        print("🤖 Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        print(f"✅ Model loaded: {self.model.__class__.__name__}")
        
        # Load audio tokenizer with CORRECT paths
        print("🎵 Loading GLM-4-Voice audio tokenizer...")
        if AUDIO_MODULES_AVAILABLE:
            try:
                self.audio_tokenizer = get_audio_tokenizer(
                    audio_tokenizer_path,
                    audio_tokenizer_type,
                    flow_path=flow_path,
                    rank=audio_tokenizer_rank,
                )
                print(f"✅ GLM-4-Voice audio tokenizer loaded: {self.audio_tokenizer.__class__.__name__}")
            except Exception as e:
                print(f"❌ Error loading GLM-4-Voice audio tokenizer: {e}")
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
        print(f"Audio offset: {self.audio_offset}")

    def run_infer(self, audio_path=None, prompt_audio_path=None, message="", task_type="Spoken QA",
                  stream_stride=4, max_returned_tokens=4096, sample_rate=16000, mode=None):
        """Main inference function with CORRECT implementation and FIXED audio token extraction"""
        
        print(f"🔍 run_infer called with:")
        print(f"   - audio_path: {audio_path}")
        print(f"   - message: {message}")
        print(f"   - task_type: {task_type}")
        
        # Prepare messages based on task type and README format
        if task_type == "TTS":
            # TTS format from README: "Convert the text to speech.\n{TEXT_TO_CONVERT}"
            messages = self.default_system_message + [
                {
                    "role": "user",
                    "content": f"Convert the text to speech.\n{message}",
                }
            ]
            print(f"🎵 TTS mode: Converting '{message}' to speech")
            
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
            print(f"🎤 ASR mode: Converting audio file '{audio_path}' to text")
            
        else:  # Spoken QA
            # Spoken QA format: just <|audio|> for audio input, or regular text
            if audio_path:
                messages = self.default_system_message + [
                    {
                        "role": "user", 
                        "content": "<|audio|>",
                    }
                ]
                print(f"🗣️ Spoken QA mode: Audio input '{audio_path}'")
            else:
                messages = self.default_system_message + [
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
                print(f"🗣️ Spoken QA mode: Text input '{message}'")

        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        print(f"Input: {self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}")

        # Handle audio input processing for contiguous codec
        audios = None
        audio_indices = None
        
        # CRITICAL FIX: For ASR and Spoken QA with audio, we need to process the audio file
        if (audio_path is not None or prompt_audio_path is not None) and self.audio_tokenizer:
            print(f"🎵 Processing audio input with tokenizer...")
            
            # Check if audio tokenizer applies to user role
            if self.audio_tokenizer.apply_to_role("user", is_contiguous=True):
                print("🎵 Using contiguous codec for audio processing")
                # Contiguous codec
                audio_paths = []
                if audio_path is not None:
                    audio_paths.append(audio_path)
                    print(f"   - Added audio_path: {audio_path}")
                if prompt_audio_path is not None:
                    audio_paths.append(prompt_audio_path)
                    print(f"   - Added prompt_audio_path: {prompt_audio_path}")
                    
                input_ids, audios, audio_indices = add_audio_input_contiguous(
                    input_ids, audio_paths, self.tokenizer, self.audio_tokenizer
                )
                print(f"   - Processed {len(audio_paths)} audio files")
                print(f"   - audios shape: {audios.shape if audios is not None else None}")
                print(f"   - audio_indices: {audio_indices}")
                
            elif self.audio_tokenizer.apply_to_role("user", is_discrete=True):
                print("🎵 Using discrete codec for audio processing")
                # Discrete codec - encode audio to tokens
                if audio_path is not None:
                    audio_tokens = self.audio_tokenizer.encode(audio_path)
                    audio_tokens_str = "".join([f"<|audio_{i}|>" for i in audio_tokens])
                    print(f"   - Encoded audio to {len(audio_tokens)} tokens")
                    
                    # Replace <|audio|> in the input with actual audio tokens
                    input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    input_text = input_text.replace("<|audio|>", f"<|begin_of_audio|>{audio_tokens_str}<|end_of_audio|>")
                    
                    # Re-tokenize with audio tokens
                    input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
                    print(f"   - Re-tokenized input with audio tokens")
            else:
                print("⚠️  Audio tokenizer doesn't apply to user role")

        # Move to device
        input_ids = input_ids.to(self.model.device)

        # Generate
        torch.cuda.synchronize()
        start = time.time()
        
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
            
        print(f"🚀 Generating with parameters: {list(generation_kwargs.keys())}")
        
        outputs = self.model.generate(**generation_kwargs)
        
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"Generation time: {end - start:.2f}s")
        
        # Decode output
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Output: {output}")

        # CRITICAL FIX: Extract ONLY assistant's audio tokens for decoding
        if task_type == "Spoken QA" and audio_path:
            # For Spoken QA with audio input, extract only assistant's audio tokens
            assistant_audio_tokens = extract_assistant_audio_tokens_only(output, self.audio_offset)
            print(f"🎯 Using ONLY assistant's {len(assistant_audio_tokens)} audio tokens for decoding")
        else:
            # For TTS and ASR, use the original method (extract all audio tokens)
            assistant_audio_tokens = []
            for token_id in outputs[0]:
                if token_id >= self.audio_offset:
                    assistant_audio_tokens.append(token_id - self.audio_offset)
            print(f"Extracted {len(assistant_audio_tokens)} audio tokens for decoding (standard method)")

        # Decode audio if we have tokens and audio tokenizer
        tts_speech = None
        if len(assistant_audio_tokens) > 0 and self.audio_tokenizer:
            try:
                print("🎵 Decoding ONLY assistant's audio tokens with GLM-4-Voice...")
                tts_speech = self.audio_tokenizer.decode(
                    assistant_audio_tokens, source_speech_16k=prompt_audio_path
                )
                print(f"✅ Audio decoded successfully! Shape: {tts_speech.shape if tts_speech is not None else 'None'}")
            except Exception as e:
                print(f"❌ Audio decoding error: {e}")
                import traceback
                traceback.print_exc()
                tts_speech = None
        elif len(assistant_audio_tokens) > 0:
            print("⚠️  Audio tokens found but no audio tokenizer available")
        
        return output, tts_speech

def _launch_demo(s2s_engine):
    def predict_chatbot(chatbot, task_history, task):
        if not task_history:
            return chatbot, task_history, None
            
        chat_query = task_history[-1][0]
        print(f"🔍 Processing query: {chat_query}")
        print(f"🔍 Query type: {type(chat_query)}")

        try:
            # CRITICAL FIX: Properly detect audio vs text input
            audio_path = None
            message = ""
            
            if isinstance(chat_query, str) and is_wav(chat_query):
                # Audio file path
                audio_path = chat_query
                message = ""
                print(f"🎵 Audio input detected: {audio_path}")
            elif isinstance(chat_query, (tuple, list)) and len(chat_query) > 0:
                # Gradio audio component returns tuple/list
                if is_wav(chat_query[0]):
                    audio_path = chat_query[0]
                    message = ""
                    print(f"🎵 Audio input detected (from tuple): {audio_path}")
                else:
                    audio_path = None
                    message = str(chat_query[0])
                    print(f"📝 Text input detected (from tuple): {message}")
            else:
                # Text input
                audio_path = None
                message = str(chat_query)
                print(f"📝 Text input detected: {message}")

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
                        print(f"✅ Audio file saved: {audio_file_path}")
                    except Exception as e:
                        print(f"❌ Error saving audio: {e}")

            # Update chat
            chatbot.append((chat_query, response_text))
            
            return chatbot, task_history, audio_file_path
            
        except Exception as e:
            error_msg = f"Error during inference: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            chatbot.append((chat_query, error_msg))
            return chatbot, task_history, None

    def predict_reset_chatbot():
        return [], []

    def predict_reset_task_history():
        return []

    with gr.Blocks(title="VITA-Audio-Boost FINAL CLEAN") as demo:
        gr.Markdown(
            """<center><font size=8>VITA-Audio-Boost FINAL CLEAN</font></center>"""
        )
        gr.Markdown(
            """<center><font size=4>Perfect text cleaning + No audio echo!</font></center>"""
        )
        gr.Markdown(
            """<center>✅ TTS: Perfect ✅ ASR: Clean transcription ✅ Spoken QA: Clean response + No echo! 🎵</center>"""
        )

        chatbot = gr.Chatbot(label="VITA-Audio-Boost", height=600, type="messages")
        
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
            print(f"🔍 add_text called with:")
            print(f"   - text: {text}")
            print(f"   - audio: {audio}")
            print(f"   - audio type: {type(audio)}")
            
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
    print("🚀 Starting VITA-Audio-Boost FINAL CLEAN Demo...")
    
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
        print("✅ S2S inference engine created successfully!")
    except Exception as e:
        print(f"❌ Error creating S2S inference engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Install soundfile if not available
    try:
        import soundfile
        print("✅ soundfile available for audio generation")
    except ImportError:
        print("⚠️  Installing soundfile for audio generation...")
        os.system("pip install soundfile")

    print("🌐 Launching FINAL CLEAN demo...")
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
        print(f"❌ Error launching demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
