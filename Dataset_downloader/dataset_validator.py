#!/usr/bin/env python3
"""
VITA-Audio Dataset Validator
Comprehensive validation tool for VITA-Audio format datasets.
Validates format compliance, audio integrity, and data quality.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import hashlib
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Comprehensive validator for VITA-Audio datasets."""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.validation_results = {
            'format_validation': {},
            'audio_validation': {},
            'content_validation': {},
            'statistics': {},
            'errors': [],
            'warnings': []
        }
        
        # Validation criteria
        self.audio_criteria = {
            'sample_rate': 16000,
            'channels': 1,
            'min_duration': 0.5,
            'max_duration': 30.0,
            'max_silence_ratio': 0.8,
            'min_rms_energy': 0.001
        }
        
        # Format specifications
        self.format_specs = {
            'asr': {
                'user_content_pattern': r'Convert the speech to text\.\n<\|audio\|>',
                'assistant_content_type': 'text_only',
                'audio_count': 1
            },
            'tts': {
                'user_content_pattern': r'Convert the text to speech\.\n.+',
                'assistant_content_type': 'audio_only',
                'audio_count': 1
            },
            'speech_qa': {
                'user_content_pattern': r'<\|audio\|>',
                'assistant_content_type': 'text_and_audio',
                'audio_count': 2
            }
        }
    
    def detect_dataset_type(self, data_file: str) -> str:
        """Detect dataset type based on content patterns."""
        with open(data_file, 'r', encoding='utf-8') as f:
            # Sample first few lines to detect pattern
            sample_lines = [f.readline() for _ in range(min(10, sum(1 for _ in f)))]
            f.seek(0)
        
        type_scores = {'asr': 0, 'tts': 0, 'speech_qa': 0}
        
        for line in sample_lines:
            if not line.strip():
                continue
            
            try:
                item = json.loads(line.strip())
                messages = item.get('messages', [])
                audios = item.get('audios', [])
                
                if len(messages) == 2:
                    user_content = messages[0].get('content', '')
                    assistant_content = messages[1].get('content', '')
                    
                    # Check patterns
                    if re.search(r'Convert the speech to text', user_content):
                        type_scores['asr'] += 1
                    elif re.search(r'Convert the text to speech', user_content):
                        type_scores['tts'] += 1
                    elif user_content.strip() == '<|audio|>':
                        type_scores['speech_qa'] += 1
                    
                    # Check audio count
                    if len(audios) == 1:
                        type_scores['asr'] += 0.5
                        type_scores['tts'] += 0.5
                    elif len(audios) == 2:
                        type_scores['speech_qa'] += 1
            
            except Exception:
                continue
        
        # Return type with highest score
        return max(type_scores, key=type_scores.get)
    
    def validate_format_compliance(self, data_file: str, dataset_type: str) -> Dict[str, Any]:
        """Validate format compliance for specific dataset type."""
        logger.info(f"Validating format compliance for {dataset_type} dataset...")
        
        results = {
            'total_samples': 0,
            'valid_samples': 0,
            'format_errors': [],
            'compliance_rate': 0.0
        }
        
        spec = self.format_specs.get(dataset_type, {})
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Validating format"), 1):
                if not line.strip():
                    continue
                
                results['total_samples'] += 1
                
                try:
                    item = json.loads(line.strip())
                    
                    # Validate required fields
                    if 'messages' not in item:
                        results['format_errors'].append(f"Line {line_num}: Missing 'messages' field")
                        continue
                    
                    if 'audios' not in item:
                        results['format_errors'].append(f"Line {line_num}: Missing 'audios' field")
                        continue
                    
                    messages = item['messages']
                    audios = item['audios']
                    
                    # Validate message structure
                    if len(messages) != 2:
                        results['format_errors'].append(f"Line {line_num}: Expected 2 messages, got {len(messages)}")
                        continue
                    
                    # Validate roles
                    if messages[0].get('role') != 'user':
                        results['format_errors'].append(f"Line {line_num}: First message should have role 'user'")
                        continue
                    
                    if messages[1].get('role') != 'assistant':
                        results['format_errors'].append(f"Line {line_num}: Second message should have role 'assistant'")
                        continue
                    
                    # Validate content patterns
                    user_content = messages[0].get('content', '')
                    assistant_content = messages[1].get('content', '')
                    
                    if dataset_type in spec:
                        # Check user content pattern
                        if 'user_content_pattern' in spec:
                            if not re.search(spec['user_content_pattern'], user_content):
                                results['format_errors'].append(f"Line {line_num}: User content doesn't match expected pattern")
                                continue
                        
                        # Check assistant content type
                        if spec['assistant_content_type'] == 'text_only':
                            if '<|audio|>' in assistant_content:
                                results['format_errors'].append(f"Line {line_num}: Assistant content should be text only")
                                continue
                        elif spec['assistant_content_type'] == 'audio_only':
                            if assistant_content.strip() != '<|audio|>':
                                results['format_errors'].append(f"Line {line_num}: Assistant content should be '<|audio|>' only")
                                continue
                        elif spec['assistant_content_type'] == 'text_and_audio':
                            if '<|audio|>' not in assistant_content:
                                results['format_errors'].append(f"Line {line_num}: Assistant content should contain '<|audio|>'")
                                continue
                        
                        # Check audio count
                        if 'audio_count' in spec:
                            if len(audios) != spec['audio_count']:
                                results['format_errors'].append(f"Line {line_num}: Expected {spec['audio_count']} audio files, got {len(audios)}")
                                continue
                    
                    results['valid_samples'] += 1
                
                except json.JSONDecodeError:
                    results['format_errors'].append(f"Line {line_num}: Invalid JSON")
                except Exception as e:
                    results['format_errors'].append(f"Line {line_num}: {str(e)}")
        
        results['compliance_rate'] = results['valid_samples'] / results['total_samples'] if results['total_samples'] > 0 else 0
        return results
    
    def validate_audio_files(self, data_file: str, audio_base_dir: str) -> Dict[str, Any]:
        """Validate audio file integrity and quality."""
        logger.info("Validating audio files...")
        
        results = {
            'total_audio_files': 0,
            'valid_audio_files': 0,
            'missing_files': [],
            'corrupted_files': [],
            'quality_issues': [],
            'audio_stats': {
                'durations': [],
                'sample_rates': [],
                'channels': [],
                'rms_energies': []
            }
        }
        
        audio_base_path = Path(audio_base_dir)
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Validating audio"), 1):
                if not line.strip():
                    continue
                
                try:
                    item = json.loads(line.strip())
                    audios = item.get('audios', [])
                    
                    for audio_path in audios:
                        results['total_audio_files'] += 1
                        full_audio_path = audio_base_path / audio_path
                        
                        # Check if file exists
                        if not full_audio_path.exists():
                            results['missing_files'].append(str(audio_path))
                            continue
                        
                        try:
                            # Load and validate audio
                            audio, sr = librosa.load(str(full_audio_path), sr=None)
                            
                            # Check basic properties
                            duration = len(audio) / sr
                            channels = 1 if audio.ndim == 1 else audio.shape[0]
                            rms_energy = librosa.feature.rms(y=audio)[0].mean()
                            
                            # Store statistics
                            results['audio_stats']['durations'].append(duration)
                            results['audio_stats']['sample_rates'].append(sr)
                            results['audio_stats']['channels'].append(channels)
                            results['audio_stats']['rms_energies'].append(rms_energy)
                            
                            # Validate against criteria
                            issues = []
                            
                            if sr != self.audio_criteria['sample_rate']:
                                issues.append(f"Sample rate {sr} != {self.audio_criteria['sample_rate']}")
                            
                            if channels != self.audio_criteria['channels']:
                                issues.append(f"Channels {channels} != {self.audio_criteria['channels']}")
                            
                            if duration < self.audio_criteria['min_duration']:
                                issues.append(f"Duration {duration:.2f}s < {self.audio_criteria['min_duration']}s")
                            
                            if duration > self.audio_criteria['max_duration']:
                                issues.append(f"Duration {duration:.2f}s > {self.audio_criteria['max_duration']}s")
                            
                            if rms_energy < self.audio_criteria['min_rms_energy']:
                                issues.append(f"Low RMS energy {rms_energy:.6f}")
                            
                            # Check for excessive silence
                            silence_ratio = self._calculate_silence_ratio(audio, sr)
                            if silence_ratio > self.audio_criteria['max_silence_ratio']:
                                issues.append(f"High silence ratio {silence_ratio:.2f}")
                            
                            if issues:
                                results['quality_issues'].append({
                                    'file': str(audio_path),
                                    'line': line_num,
                                    'issues': issues
                                })
                            else:
                                results['valid_audio_files'] += 1
                        
                        except Exception as e:
                            results['corrupted_files'].append({
                                'file': str(audio_path),
                                'error': str(e)
                            })
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Error processing line {line_num}: {e}")
        
        return results
    
    def _calculate_silence_ratio(self, audio: any, sr: int, threshold: float = 0.01) -> float:
        """Calculate ratio of silence in audio."""
        # Simple silence detection based on RMS energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        silence_frames = (rms < threshold).sum()
        total_frames = len(rms)
        
        return silence_frames / total_frames if total_frames > 0 else 1.0
    
    def validate_content_quality(self, data_file: str) -> Dict[str, Any]:
        """Validate content quality (text, language, etc.)."""
        logger.info("Validating content quality...")
        
        results = {
            'text_stats': {
                'lengths': [],
                'languages': {},
                'empty_texts': 0,
                'special_chars': 0
            },
            'content_issues': []
        }
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Validating content"), 1):
                if not line.strip():
                    continue
                
                try:
                    item = json.loads(line.strip())
                    messages = item.get('messages', [])
                    
                    for msg in messages:
                        content = msg.get('content', '')
                        
                        # Skip audio tokens
                        if content.strip() == '<|audio|>':
                            continue
                        
                        # Extract text content (remove instructions)
                        text_content = content
                        if content.startswith('Convert the speech to text.'):
                            continue  # Skip instruction
                        elif content.startswith('Convert the text to speech.'):
                            text_content = content.replace('Convert the text to speech.\n', '')
                        
                        if text_content and text_content != '<|audio|>':
                            # Text length
                            results['text_stats']['lengths'].append(len(text_content))
                            
                            # Check for empty text
                            if not text_content.strip():
                                results['text_stats']['empty_texts'] += 1
                            
                            # Check for special characters
                            if re.search(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff.,!?;:\'"()-]', text_content):
                                results['text_stats']['special_chars'] += 1
                            
                            # Simple language detection (basic heuristic)
                            if re.search(r'[\u4e00-\u9fff]', text_content):
                                lang = 'chinese'
                            elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text_content):
                                lang = 'japanese'
                            elif re.search(r'[a-zA-Z]', text_content):
                                lang = 'latin_script'
                            else:
                                lang = 'other'
                            
                            results['text_stats']['languages'][lang] = results['text_stats']['languages'].get(lang, 0) + 1
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Error processing line {line_num}: {e}")
        
        return results
    
    def generate_statistics(self, format_results: Dict, audio_results: Dict, content_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive statistics."""
        stats = {}
        
        # Format statistics
        stats['format'] = {
            'total_samples': format_results.get('total_samples', 0),
            'valid_samples': format_results.get('valid_samples', 0),
            'compliance_rate': format_results.get('compliance_rate', 0),
            'error_count': len(format_results.get('format_errors', []))
        }
        
        # Audio statistics
        if audio_results['audio_stats']['durations']:
            durations = audio_results['audio_stats']['durations']
            stats['audio'] = {
                'total_files': audio_results['total_audio_files'],
                'valid_files': audio_results['valid_audio_files'],
                'missing_files': len(audio_results['missing_files']),
                'corrupted_files': len(audio_results['corrupted_files']),
                'quality_issues': len(audio_results['quality_issues']),
                'duration_stats': {
                    'mean': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'total_hours': sum(durations) / 3600
                }
            }
        
        # Content statistics
        if content_results['text_stats']['lengths']:
            lengths = content_results['text_stats']['lengths']
            stats['content'] = {
                'text_samples': len(lengths),
                'empty_texts': content_results['text_stats']['empty_texts'],
                'special_chars': content_results['text_stats']['special_chars'],
                'languages': content_results['text_stats']['languages'],
                'text_length_stats': {
                    'mean': sum(lengths) / len(lengths),
                    'min': min(lengths),
                    'max': max(lengths)
                }
            }
        
        return stats
    
    def validate_dataset(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Validate complete dataset."""
        logger.info(f"Starting validation of dataset in {self.dataset_dir}")
        
        # Find data files
        data_files = list(self.dataset_dir.rglob("data.jsonl"))
        
        if not data_files:
            logger.error("No data.jsonl files found")
            return {'error': 'No data files found'}
        
        all_results = {}
        
        for data_file in data_files:
            task_dir = data_file.parent
            task_name = task_dir.name
            
            logger.info(f"Validating task: {task_name}")
            
            # Detect or use provided task type
            if task_type:
                detected_type = task_type
            else:
                detected_type = self.detect_dataset_type(str(data_file))
            
            logger.info(f"Detected/using task type: {detected_type}")
            
            # Validate format
            format_results = self.validate_format_compliance(str(data_file), detected_type)
            
            # Validate audio files
            audio_dir = task_dir / "audio"
            if audio_dir.exists():
                audio_results = self.validate_audio_files(str(data_file), str(audio_dir))
            else:
                audio_results = {'error': 'Audio directory not found'}
            
            # Validate content
            content_results = self.validate_content_quality(str(data_file))
            
            # Generate statistics
            stats = self.generate_statistics(format_results, audio_results, content_results)
            
            all_results[task_name] = {
                'task_type': detected_type,
                'format_validation': format_results,
                'audio_validation': audio_results,
                'content_validation': content_results,
                'statistics': stats
            }
        
        return all_results
    
    def generate_validation_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate comprehensive validation report."""
        report_lines = []
        
        # Header
        report_lines.extend([
            "=" * 80,
            "VITA-AUDIO DATASET VALIDATION REPORT",
            "=" * 80,
            f"Dataset Directory: {self.dataset_dir}",
            f"Validation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        # Summary
        total_samples = sum(r.get('statistics', {}).get('format', {}).get('total_samples', 0) for r in results.values())
        total_valid = sum(r.get('statistics', {}).get('format', {}).get('valid_samples', 0) for r in results.values())
        
        report_lines.extend([
            "VALIDATION SUMMARY:",
            "-" * 20,
            f"Total tasks validated: {len(results)}",
            f"Total samples: {total_samples:,}",
            f"Valid samples: {total_valid:,}",
            f"Overall compliance rate: {(total_valid/total_samples*100):.1f}%" if total_samples > 0 else "N/A",
            ""
        ])
        
        # Task-specific results
        for task_name, task_results in results.items():
            if 'error' in task_results:
                report_lines.extend([
                    f"TASK: {task_name} - ERROR",
                    f"Error: {task_results['error']}",
                    ""
                ])
                continue
            
            task_type = task_results.get('task_type', 'unknown')
            stats = task_results.get('statistics', {})
            
            report_lines.extend([
                f"TASK: {task_name} ({task_type.upper()})",
                "=" * 50
            ])
            
            # Format validation
            format_stats = stats.get('format', {})
            report_lines.extend([
                "Format Validation:",
                f"  Total samples: {format_stats.get('total_samples', 0):,}",
                f"  Valid samples: {format_stats.get('valid_samples', 0):,}",
                f"  Compliance rate: {format_stats.get('compliance_rate', 0)*100:.1f}%",
                f"  Format errors: {format_stats.get('error_count', 0)}"
            ])
            
            # Audio validation
            audio_stats = stats.get('audio', {})
            if audio_stats:
                duration_stats = audio_stats.get('duration_stats', {})
                report_lines.extend([
                    "",
                    "Audio Validation:",
                    f"  Total audio files: {audio_stats.get('total_files', 0):,}",
                    f"  Valid files: {audio_stats.get('valid_files', 0):,}",
                    f"  Missing files: {audio_stats.get('missing_files', 0)}",
                    f"  Corrupted files: {audio_stats.get('corrupted_files', 0)}",
                    f"  Quality issues: {audio_stats.get('quality_issues', 0)}",
                    f"  Total duration: {duration_stats.get('total_hours', 0):.1f} hours",
                    f"  Average duration: {duration_stats.get('mean', 0):.1f} seconds"
                ])
            
            # Content validation
            content_stats = stats.get('content', {})
            if content_stats:
                report_lines.extend([
                    "",
                    "Content Validation:",
                    f"  Text samples: {content_stats.get('text_samples', 0):,}",
                    f"  Empty texts: {content_stats.get('empty_texts', 0)}",
                    f"  Languages detected: {list(content_stats.get('languages', {}).keys())}"
                ])
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 20
        ])
        
        for task_name, task_results in results.items():
            if 'error' in task_results:
                continue
            
            format_results = task_results.get('format_validation', {})
            audio_results = task_results.get('audio_validation', {})
            
            if format_results.get('compliance_rate', 0) < 0.95:
                report_lines.append(f"• {task_name}: Fix format compliance issues")
            
            if audio_results.get('missing_files'):
                report_lines.append(f"• {task_name}: Resolve missing audio files")
            
            if audio_results.get('quality_issues'):
                report_lines.append(f"• {task_name}: Address audio quality issues")
        
        if not any('•' in line for line in report_lines[-10:]):
            report_lines.append("• All datasets passed validation!")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Validation report saved to {output_file}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Validate VITA-Audio datasets")
    parser.add_argument("dataset_dir", help="Directory containing converted datasets")
    parser.add_argument("--task-type", choices=["asr", "tts", "speech_qa"], help="Force specific task type")
    parser.add_argument("--output", help="Output file for validation report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = DatasetValidator(args.dataset_dir)
    
    # Run validation
    results = validator.validate_dataset(args.task_type)
    
    # Generate report
    report = validator.generate_validation_report(results, args.output)
    
    if not args.output:
        print(report)

if __name__ == "__main__":
    main()
