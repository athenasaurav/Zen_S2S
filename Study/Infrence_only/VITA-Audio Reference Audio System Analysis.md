# VITA-Audio Reference Audio System Analysis

## Key Discovery: "Your Voice" System Message

### The Critical Code Section

From `zen-vita-audio/tools/inference_sts.py`:

```python
if prompt_audio_path is not None:
    system_message = [
        {
            "role": "system",
            "content": f"Your Voice: <|audio|>\n",
        },
    ]
```

### What This Means

**The "Your Voice: <|audio|>" is a system message that provides reference audio to VITA-Audio to influence the voice characteristics of the response.**

## How the Reference Audio System Works

### Step 1: Audio Reference Injection
```python
if prompt_audio_path is not None and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
    # discrete codec
    audio_tokens = self.audio_tokenizer.encode(prompt_audio_path)
    audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
    system_message[-1]["content"] = system_message[-1]["content"].replace(
        "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
    )
```

**Process:**
1. `prompt_audio_path` contains the reference audio file
2. Audio is tokenized using the audio tokenizer
3. Tokens are formatted as `<|audio_0|><|audio_1|>...`
4. The `<|audio|>` placeholder is replaced with actual audio tokens
5. Final system message becomes: `"Your Voice: <|begin_of_audio|><|audio_0|><|audio_1|>...<|end_of_audio|>"`

### Step 2: Voice Cloning Usage Examples

From the code, we can see practical examples:

```python
for prompt_audio_path in [
    "asset/2631296891109983590.wav",
    "asset/379838640-d5ff0815-74f8-4738-b0f1-477cfc8dcc2d.wav", 
    "asset/4202818730519913143.wav",
]:
    output, tts_speech = s2s_inference.run_infer(
        prompt_audio_path=prompt_audio_path,
        message="Convert the text to speech.\n" + text,
        mode=None,
        do_sample=True,
    )
```

**This shows:**
- Multiple reference audio files are used for voice cloning
- The system can clone different voices based on the reference audio
- The reference audio is passed as `prompt_audio_path`

## The Complete Audio Flow

### Input Audio vs Reference Audio

1. **Input Audio** (`audio_path`): The user's current speech input
2. **Reference Audio** (`prompt_audio_path`): The voice sample to clone/mimic

### Message Construction

```python
if audio_path is not None:
    messages = system_message + [
        {
            "role": "user", 
            "content": message + "\n<|audio|>",
        },
    ]
else:
    messages = system_message + [
        {
            "role": "user",
            "content": message,
        },
    ]
```

**Complete conversation structure:**
```
System: "Your Voice: <|begin_of_audio|>[reference_audio_tokens]<|end_of_audio|>"
User: "[text_message]\n<|audio|>" (where <|audio|> gets replaced with input audio tokens)
```

## Key Insights

### 1. Dual Audio System
- **Reference Audio**: Defines the target voice characteristics (via system message)
- **Input Audio**: The user's current speech (via user message)

### 2. Voice Cloning Mechanism
- The system message with reference audio acts as a "voice template"
- VITA-Audio learns to respond using the voice characteristics from the reference audio
- This enables zero-shot voice cloning capabilities

### 3. Tokenization Process
- Both reference and input audio are tokenized using the same audio tokenizer
- Discrete tokens are wrapped with special markers (`<|begin_of_audio|>`, `<|end_of_audio|>`)
- The model understands these tokens as audio representations


## Web Demo Implementations Analysis

### 1. web_demo.py - Basic Web Interface

#### Audio Interface Components
```python
record_btn = gr.Audio(
    sources=["microphone", "upload"],
    type="filepath", 
    label="🎤 Record or Upload Audio",
    show_download_button=True,
    waveform_options=gr.WaveformOptions(sample_rate=16000),
)
audio_output = gr.Audio(
    label="Play", streaming=True, autoplay=True, show_download_button=True
)
```

**Key Features:**
- **Input**: Microphone recording OR file upload
- **Output**: Streaming audio with autoplay
- **Sample Rate**: 16kHz (standard for speech processing)
- **No explicit reference audio interface** - uses default voice characteristics

### 2. web_demo_stream.py - Streaming Interface with Reference Audio

#### Reference Audio System
```python
if prompt_audio_path is not None:
    if audio_tokenizer.apply_to_role("system", is_discrete=True):
        prompt_audio_tokens = audio_tokenizer.encode(prompt_audio_path)
        prompt_audio_tokens = "".join(f"<|audio_{i}|>" for i in prompt_audio_tokens)
        system_message = [
            {
                "role": "system",
                "content": f"Your Voice: <|begin_of_audio|>{prompt_audio_tokens}<|end_of_audio|>\n",
            },
        ]
```

#### Audio Decoding with Reference
```python
tts_speech = audio_tokenizer.decode(
    audio_tokens,
    source_speech_16k=prompt_audio_path,  # Reference audio for voice cloning
    option_steps=option_steps,
)
```

**Key Features:**
- **Reference Audio Support**: Uses `prompt_audio_path` for voice cloning
- **System Message Integration**: Reference audio embedded in system prompt
- **Streaming Synthesis**: Real-time audio generation with reference voice
- **Voice Cloning**: `source_speech_16k` parameter enables voice characteristic transfer

### 3. web_demo_stream_local.py - Local Streaming with Reference Audio

#### Reference Audio Configuration
```python
prompt_audio_path = None  # Can be set to enable voice cloning

if prompt_audio_path is not None:
    if audio_tokenizer.apply_to_role("system", is_discrete=True):
        prompt_audio_tokens = audio_tokenizer.encode(prompt_audio_path)
        prompt_audio_tokens = "".join(f"<|audio_{i}|>" for i in prompt_audio_tokens)
        system_message = [
            {
                "role": "system", 
                "content": f"Your Voice: <|begin_of_audio|>{prompt_audio_tokens}<|end_of_audio|>\n",
            },
        ]
```

#### Local Audio Processing
```python
tts_speech = audio_tokenizer.decode(
    audio_tokens,
    source_speech_16k=prompt_audio_path,
    option_steps=option_steps,
)
```

**Key Features:**
- **Local Processing**: No external API dependencies
- **Configurable Reference**: `prompt_audio_path` can be set programmatically
- **Same Voice Cloning Logic**: Identical to streaming version
- **Optimized for Local Deployment**: Reduced network dependencies

## Complete Audio Flow Analysis

### The Dual Audio System Revealed

#### 1. Input Audio Flow
```
User Input Audio → Audio Tokenizer → User Message Tokens
```

#### 2. Reference Audio Flow  
```
Reference Audio File → Audio Tokenizer → System Message Tokens
```

#### 3. Combined Message Structure
```
System: "Your Voice: <|begin_of_audio|>[reference_tokens]<|end_of_audio|>"
User: "[text_input]\n<|audio|>" (where <|audio|> = input audio tokens)
```

#### 4. Response Generation
```
Combined Messages → VITA-Audio Model → Response Tokens (Text + Audio)
```

#### 5. Audio Synthesis with Voice Cloning
```
Audio Response Tokens + Reference Audio → GLM-4-Voice Decoder → Final Audio
```

## Key Technical Insights

### 1. Reference Audio Sources

**From the code analysis:**
- **Hardcoded in inference_sts.py**: Uses asset files like `"asset/2631296891109983590.wav"`
- **Configurable in web demos**: Can be set via `prompt_audio_path` parameter
- **User uploadable**: Web interface allows users to provide reference audio

### 2. Voice Cloning Mechanism

**Two-stage process:**
1. **System Message**: Reference audio tokens tell the model "this is the target voice"
2. **Audio Synthesis**: `source_speech_16k` parameter provides voice characteristics to decoder

### 3. Audio Tokenization Process

```python
# Encode reference audio to discrete tokens
prompt_audio_tokens = audio_tokenizer.encode(prompt_audio_path)
prompt_audio_tokens = "".join(f"<|audio_{i}|>" for i in prompt_audio_tokens)

# Embed in system message
system_content = f"Your Voice: <|begin_of_audio|>{prompt_audio_tokens}<|end_of_audio|>\n"
```

**Process:**
1. Reference audio → Discrete tokens (e.g., [1, 45, 123, 67, ...])
2. Format as token sequence: `<|audio_1|><|audio_45|><|audio_123|><|audio_67|>...`
3. Wrap with markers: `<|begin_of_audio|>...<|end_of_audio|>`
4. Embed in system message: `"Your Voice: <|begin_of_audio|>...<|end_of_audio|>"`

### 4. Streaming vs Non-Streaming Differences

#### Non-Streaming (web_demo.py)
- **Simple interface**: Single audio input/output
- **No explicit reference audio**: Uses model's default voice characteristics
- **Batch processing**: Complete response generated before playback

#### Streaming (web_demo_stream.py, web_demo_stream_local.py)
- **Reference audio support**: Explicit voice cloning capabilities
- **Real-time synthesis**: Audio generated as tokens are produced
- **Progressive steps**: `option_steps` parameter for quality vs speed trade-off

## Where Audio Comes From

### 1. Reference Audio (`prompt_audio_path`)
**Sources:**
- **Asset files**: Pre-recorded voice samples in `asset/` directory
- **User uploads**: Via web interface file upload
- **Programmatic**: Set directly in code configuration
- **Microphone**: Can record reference voice sample

**Purpose:**
- Defines target voice characteristics
- Enables zero-shot voice cloning
- Provides voice style template

### 2. Input Audio (`audio_path`)
**Sources:**
- **Microphone recording**: Real-time user speech
- **File upload**: Pre-recorded user audio
- **Streaming input**: Continuous audio stream

**Purpose:**
- Contains user's actual speech content
- Provides conversation context
- May influence response voice characteristics

### 3. Output Audio
**Generated by:**
- **GLM-4-Voice Decoder**: Converts response tokens to speech tokens
- **CosyVoice**: Final audio synthesis with voice characteristics
- **Reference audio influence**: Voice cloning from `source_speech_16k`

**Characteristics determined by:**
- Reference audio voice style
- User input voice characteristics  
- Model learned patterns
- Explicit voice instructions in text

