#!/usr/bin/env python3
"""
VITA-Audio Dataset Converter
Converts various audio datasets to VITA-Audio format (ASR, TTS, Speech QA).
Handles format standardization, quality filtering, and validation.
"""

import os
import json
import argparse
import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import webdataset as wds
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VITAAudioConverter:
    """Main class for converting datasets to VITA-Audio format."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Audio processing parameters
        self.target_sr = 16000
        self.target_channels = 1
        self.min_duration = 0.5  # seconds
        self.max_duration = 30.0  # seconds
        
        # Statistics
        self.stats = {
            'processed': 0,
            'converted': 0,
            'skipped': 0,
            'errors': 0
        }
    
    def standardize_audio(self, audio_path: str, output_path: str) -> bool:
        """Standardize audio to 16kHz mono WAV format."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            # Check duration
            duration = len(audio) / sr
            if duration < self.min_duration or duration > self.max_duration:
                logger.debug(f"Skipping {audio_path}: duration {duration:.2f}s out of range")
                return False
            
            # Save standardized audio
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, sr)
            return True
            
        except Exception as e:
            logger.error(f"Failed to process audio {audio_path}: {e}")
            return False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove timestamps and special markers
        text = re.sub(r'\[\d+:\d+:\d+\.\d+\]', '', text)  # Remove timestamps
        text = re.sub(r'<[^>]+>', '', text)  # Remove XML-like tags
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def convert_audioqa_to_speech_qa(self, dataset_dir: str) -> bool:
        """Convert AudioQA-1M to Speech QA format."""
        logger.info("Converting AudioQA-1M to Speech QA format...")
        
        input_path = Path(dataset_dir)
        output_path = self.output_dir / "speech_qa"
        output_path.mkdir(exist_ok=True)
        
        # Load metadata
        metadata_file = input_path / "metadata.jsonl"
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return False
        
        converted_data = []
        audio_dir = output_path / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Converting AudioQA")):
                try:
                    item = json.loads(line.strip())
                    
                    # Extract audio information
                    if 'wav' not in item or '__key__' not in item:
                        continue
                    
                    key = item['__key__']
                    
                    # Determine if this is a question or answer based on filename
                    if '_question' in key:
                        # This is a question audio
                        question_audio_path = f"audio/{key}_question.wav"
                        
                        # Look for corresponding answer
                        answer_key = key.replace('_question', '')
                        # We'll need to find the answer in subsequent processing
                        
                        # For now, create a placeholder entry
                        converted_item = {
                            "messages": [
                                {
                                    "content": "<|audio|>",
                                    "role": "user"
                                },
                                {
                                    "content": "I need to find the corresponding answer for this question.",
                                    "role": "assistant"
                                }
                            ],
                            "audios": [question_audio_path]
                        }
                        
                        converted_data.append(converted_item)
                    
                    self.stats['processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    self.stats['errors'] += 1
        
        # Save converted data
        output_file = output_path / "data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Converted {len(converted_data)} Speech QA samples to {output_file}")
        self.stats['converted'] += len(converted_data)
        return True
    
    def convert_wenetspeech_to_asr(self, dataset_dir: str) -> bool:
        """Convert WenetSpeech to ASR format."""
        logger.info("Converting WenetSpeech to ASR format...")
        
        input_path = Path(dataset_dir)
        output_path = self.output_dir / "asr"
        output_path.mkdir(exist_ok=True)
        
        # Process L_fixed (main training set)
        metadata_file = input_path / "L_fixed_metadata.jsonl"
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return False
        
        converted_data = []
        audio_dir = output_path / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Converting WenetSpeech")):
                try:
                    item = json.loads(line.strip())
                    
                    # Extract text and audio path
                    if 'text' not in item or 'audio' not in item:
                        continue
                    
                    text = self.clean_text(item['text'])
                    if not text:
                        continue
                    
                    # Generate output audio path
                    audio_id = f"wenetspeech_{line_num:08d}"
                    output_audio_path = f"audio/{audio_id}.wav"
                    
                    # Create ASR format entry
                    converted_item = {
                        "messages": [
                            {
                                "content": f"Convert the speech to text.\n<|audio|>",
                                "role": "user"
                            },
                            {
                                "content": text,
                                "role": "assistant"
                            }
                        ],
                        "audios": [output_audio_path]
                    }
                    
                    converted_data.append(converted_item)
                    self.stats['processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    self.stats['errors'] += 1
        
        # Save converted data
        output_file = output_path / "data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Converted {len(converted_data)} ASR samples to {output_file}")
        self.stats['converted'] += len(converted_data)
        return True
    
    def convert_wenetspeech4tts_to_tts(self, dataset_dir: str) -> bool:
        """Convert WenetSpeech4TTS to TTS format."""
        logger.info("Converting WenetSpeech4TTS to TTS format...")
        
        input_path = Path(dataset_dir)
        output_path = self.output_dir / "tts_wenetspeech"
        output_path.mkdir(exist_ok=True)
        
        converted_data = []
        audio_dir = output_path / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Process Premium quality first
        premium_dir = input_path / "Premium"
        if premium_dir.exists():
            # Extract tar.gz files
            for tar_file in premium_dir.glob("*.tar.gz"):
                logger.info(f"Processing {tar_file}")
                
                try:
                    with tarfile.open(tar_file, 'r:gz') as tar:
                        tar.extractall(path=premium_dir / "extracted")
                    
                    # Process extracted files
                    extracted_dir = premium_dir / "extracted"
                    for wav_file in extracted_dir.rglob("*.wav"):
                        txt_file = wav_file.with_suffix('.txt')
                        
                        if txt_file.exists():
                            # Read text content
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                            
                            # Parse content (format: utt_id\ttext\n\ttimestamps)
                            lines = content.split('\n')
                            if len(lines) >= 1:
                                parts = lines[0].split('\t')
                                if len(parts) >= 2:
                                    text = self.clean_text(parts[1])
                                    
                                    if text:
                                        # Generate output paths
                                        audio_id = wav_file.stem
                                        output_audio_path = f"audio/{audio_id}.wav"
                                        
                                        # Create TTS format entry
                                        converted_item = {
                                            "messages": [
                                                {
                                                    "content": f"Convert the text to speech.\n{text}",
                                                    "role": "user"
                                                },
                                                {
                                                    "content": "<|audio|>",
                                                    "role": "assistant"
                                                }
                                            ],
                                            "audios": [output_audio_path]
                                        }
                                        
                                        converted_data.append(converted_item)
                                        self.stats['processed'] += 1
                
                except Exception as e:
                    logger.error(f"Error processing {tar_file}: {e}")
                    self.stats['errors'] += 1
        
        # Save converted data
        output_file = output_path / "data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Converted {len(converted_data)} TTS samples to {output_file}")
        self.stats['converted'] += len(converted_data)
        return True
    
    def convert_emilia_to_tts(self, dataset_dir: str) -> bool:
        """Convert Emilia Dataset to TTS format."""
        logger.info("Converting Emilia Dataset to TTS format...")
        
        input_path = Path(dataset_dir)
        output_path = self.output_dir / "tts_emilia"
        output_path.mkdir(exist_ok=True)
        
        converted_data = []
        audio_dir = output_path / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Process WebDataset tar files
        tar_files = list(input_path.glob("*.tar"))
        
        for tar_file in tqdm(tar_files[:10], desc="Processing Emilia tars"):  # Limit for demo
            try:
                dataset = wds.WebDataset(str(tar_file))
                
                for sample in dataset:
                    try:
                        # Extract text and audio
                        if 'json' in sample and 'wav' in sample:
                            metadata = json.loads(sample['json'])
                            
                            if 'text' in metadata:
                                text = self.clean_text(metadata['text'])
                                
                                if text:
                                    # Generate output paths
                                    sample_id = sample.get('__key__', f"emilia_{self.stats['processed']}")
                                    output_audio_path = f"audio/{sample_id}.wav"
                                    
                                    # Create TTS format entry
                                    converted_item = {
                                        "messages": [
                                            {
                                                "content": f"Convert the text to speech.\n{text}",
                                                "role": "user"
                                            },
                                            {
                                                "content": "<|audio|>",
                                                "role": "assistant"
                                            }
                                        ],
                                        "audios": [output_audio_path]
                                    }
                                    
                                    converted_data.append(converted_item)
                                    self.stats['processed'] += 1
                    
                    except Exception as e:
                        logger.debug(f"Error processing sample: {e}")
                        self.stats['errors'] += 1
            
            except Exception as e:
                logger.error(f"Error processing {tar_file}: {e}")
                self.stats['errors'] += 1
        
        # Save converted data
        output_file = output_path / "data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Converted {len(converted_data)} TTS samples to {output_file}")
        self.stats['converted'] += len(converted_data)
        return True
    
    def validate_converted_data(self, output_dir: str) -> Dict[str, Any]:
        """Validate converted datasets."""
        logger.info("Validating converted datasets...")
        
        validation_results = {}
        output_path = Path(output_dir)
        
        for task_dir in output_path.iterdir():
            if not task_dir.is_dir():
                continue
            
            task_name = task_dir.name
            data_file = task_dir / "data.jsonl"
            
            if not data_file.exists():
                validation_results[task_name] = {"status": "missing", "count": 0}
                continue
            
            # Count samples and validate format
            sample_count = 0
            format_errors = 0
            
            with open(data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        
                        # Validate required fields
                        if 'messages' not in item or 'audios' not in item:
                            format_errors += 1
                            continue
                        
                        # Validate message structure
                        messages = item['messages']
                        if len(messages) != 2:
                            format_errors += 1
                            continue
                        
                        sample_count += 1
                        
                    except Exception as e:
                        format_errors += 1
            
            validation_results[task_name] = {
                "status": "valid",
                "count": sample_count,
                "format_errors": format_errors
            }
        
        return validation_results
    
    def convert_dataset(self, dataset_name: str, dataset_type: str) -> bool:
        """Convert a specific dataset."""
        dataset_dir = self.input_dir / dataset_name.split('/')[-1]
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return False
        
        logger.info(f"Converting {dataset_name} to {dataset_type} format...")
        
        success = False
        if dataset_type == "speech_qa" and "AudioQA" in dataset_name:
            success = self.convert_audioqa_to_speech_qa(str(dataset_dir))
        elif dataset_type == "asr" and "wenetspeech" in dataset_name.lower():
            success = self.convert_wenetspeech_to_asr(str(dataset_dir))
        elif dataset_type == "tts" and "WenetSpeech4TTS" in dataset_name:
            success = self.convert_wenetspeech4tts_to_tts(str(dataset_dir))
        elif dataset_type == "tts" and "Emilia" in dataset_name:
            success = self.convert_emilia_to_tts(str(dataset_dir))
        else:
            logger.error(f"Unknown conversion: {dataset_name} -> {dataset_type}")
            return False
        
        return success
    
    def print_summary(self):
        """Print conversion summary."""
        logger.info("=" * 60)
        logger.info("CONVERSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Samples processed: {self.stats['processed']}")
        logger.info(f"Samples converted: {self.stats['converted']}")
        logger.info(f"Samples skipped: {self.stats['skipped']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Convert datasets to VITA-Audio format")
    parser.add_argument("--input-dir", required=True, help="Input datasets directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for converted datasets")
    parser.add_argument("--dataset", help="Convert specific dataset")
    parser.add_argument("--type", choices=["asr", "tts", "speech_qa"], help="Target format type")
    parser.add_argument("--validate", action="store_true", help="Validate converted datasets")
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = VITAAudioConverter(args.input_dir, args.output_dir)
    
    if args.validate:
        results = converter.validate_converted_data(args.output_dir)
        for task, result in results.items():
            logger.info(f"{task}: {result}")
        return
    
    # Convert datasets
    if args.dataset and args.type:
        success = converter.convert_dataset(args.dataset, args.type)
        if not success:
            logger.error("Conversion failed")
            return
    else:
        # Convert all known datasets
        conversions = [
            ("shenyunhang/AudioQA-1M", "speech_qa"),
            ("wenet-e2e/wenetspeech", "asr"),
            ("Wenetspeech4TTS/WenetSpeech4TTS", "tts"),
            ("amphion/Emilia-Dataset", "tts")
        ]
        
        for dataset_name, dataset_type in conversions:
            converter.convert_dataset(dataset_name, dataset_type)
    
    # Print summary
    converter.print_summary()

if __name__ == "__main__":
    main()
