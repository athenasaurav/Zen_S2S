#!/usr/bin/env python3
"""
VITA-Audio Dataset Downloader
Comprehensive script to download and prepare all datasets for VITA-Audio training.
Supports ASR, TTS, and Speech QA datasets with proper format conversion.
Fixed to handle wenet-e2e/wenetspeech by downloading compressed JSONL files directly.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import tarfile
import zipfile
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import requests
from tqdm import tqdm
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Main class for downloading and converting VITA-Audio datasets."""
    
    def __init__(self, base_dir: str = "./datasets", hf_token: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.hf_token = hf_token
        self.stats = {
            'downloaded': 0,
            'converted': 0,
            'errors': 0,
            'total_size': 0
        }
        
        # Dataset configurations
        self.datasets = {
            'shenyunhang/AudioQA-1M': {
                'type': 'speech_qa',
                'priority': 1,
                'size_gb': 70,
                'requires_approval': False,
                'format': 'webdataset'
            },
            'wenet-e2e/wenetspeech': {
                'type': 'asr',
                'priority': 1,
                'size_gb': 500,
                'requires_approval': True,
                'format': 'compressed_jsonl'
            },
            'Wenetspeech4TTS/WenetSpeech4TTS': {
                'type': 'tts',
                'priority': 2,
                'size_gb': 200,
                'requires_approval': True,
                'format': 'tar_gz'
            },
            'amphion/Emilia-Dataset': {
                'type': 'tts',
                'priority': 3,
                'size_gb': 4500,
                'requires_approval': True,
                'format': 'webdataset'
            }
        }
    
    def check_requirements(self) -> bool:
        """Check if all required tools and libraries are available."""
        logger.info("Checking requirements...")
        
        required_packages = [
            'datasets', 'huggingface_hub', 'librosa', 'soundfile', 
            'webdataset', 'torchaudio', 'transformers'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        # Check disk space
        total_size = sum(ds['size_gb'] for ds in self.datasets.values())
        available_space = self._get_available_space()
        
        if available_space < total_size * 1.5:  # 1.5x for processing space
            logger.warning(f"Low disk space. Need ~{total_size * 1.5:.1f}GB, have {available_space:.1f}GB")
        
        return True
    
    def _get_available_space(self) -> float:
        """Get available disk space in GB."""
        statvfs = os.statvfs(self.base_dir)
        return (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    
    def setup_huggingface(self) -> bool:
        """Setup HuggingFace authentication."""
        if not self.hf_token:
            logger.warning("No HuggingFace token provided. Some datasets may not be accessible.")
            return False
        
        try:
            from huggingface_hub import login
            login(token=self.hf_token)
            logger.info("HuggingFace authentication successful")
            return True
        except Exception as e:
            logger.error(f"HuggingFace authentication failed: {e}")
            return False
    
    def download_audioqa_1m(self) -> bool:
        """Download shenyunhang/AudioQA-1M dataset."""
        logger.info("Downloading AudioQA-1M dataset...")
        dataset_dir = self.base_dir / "AudioQA-1M"
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            from datasets import load_dataset
            
            # Download the dataset
            dataset = load_dataset("shenyunhang/AudioQA-1M", streaming=False)
            
            # Save metadata
            metadata_file = dataset_dir / "metadata.jsonl"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                for item in tqdm(dataset['train'], desc="Processing AudioQA-1M"):
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"AudioQA-1M metadata saved to {metadata_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download AudioQA-1M: {e}")
            return False
    
    def download_wenetspeech(self) -> bool:
        """Download wenet-e2e/wenetspeech dataset by downloading compressed JSONL files directly."""
        logger.info("Downloading WenetSpeech dataset...")
        dataset_dir = self.base_dir / "wenetspeech"
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            # List all files in the repository
            logger.info("Listing WenetSpeech repository files...")
            repo_files = list_repo_files("wenet-e2e/wenetspeech", repo_type="dataset", token=self.hf_token)
            
            # Filter for the JSONL.GZ files we need
            target_patterns = [
                "cuts_L_fixed",      # Main training set
                "cuts_DEV_fixed",    # Development set
                "cuts_TEST_NET",     # Test set (NET)
                "cuts_TEST_MEETING"  # Test set (MEETING)
            ]
            
            jsonl_files = []
            for file_path in repo_files:
                if file_path.startswith("data/") and file_path.endswith(".jsonl.gz"):
                    for pattern in target_patterns:
                        if pattern in file_path:
                            jsonl_files.append(file_path)
                            break
            
            logger.info(f"Found {len(jsonl_files)} JSONL files to download")
            
            # Download and process each JSONL.GZ file
            downloaded_configs = set()
            
            for jsonl_file in tqdm(jsonl_files, desc="Downloading WenetSpeech files"):
                try:
                    logger.info(f"Downloading {jsonl_file}...")
                    
                    # Download the compressed file
                    local_path = hf_hub_download(
                        repo_id="wenet-e2e/wenetspeech",
                        filename=jsonl_file,
                        repo_type="dataset",
                        token=self.hf_token,
                        local_dir=str(dataset_dir / "raw")
                    )
                    
                    # Extract configuration name from filename
                    filename = Path(jsonl_file).name
                    if "L_fixed" in filename:
                        config_name = "L_fixed"
                    elif "DEV_fixed" in filename:
                        config_name = "DEV_fixed"
                    elif "TEST_NET" in filename:
                        config_name = "TEST_NET"
                    elif "TEST_MEETING" in filename:
                        config_name = "TEST_MEETING"
                    else:
                        continue
                    
                    # Decompress and merge files for each configuration
                    output_file = dataset_dir / f"{config_name}_metadata.jsonl"
                    
                    # Read compressed JSONL and append to output file
                    with gzip.open(local_path, 'rt', encoding='utf-8') as gz_file:
                        with open(output_file, 'a', encoding='utf-8') as out_file:
                            line_count = 0
                            for line in gz_file:
                                out_file.write(line)
                                line_count += 1
                    
                    downloaded_configs.add(config_name)
                    logger.info(f"Processed {jsonl_file} -> {config_name}_metadata.jsonl ({line_count} lines)")
                    
                except Exception as e:
                    logger.error(f"Failed to download {jsonl_file}: {e}")
                    continue
            
            # Verify we got the essential files
            if "L_fixed" in downloaded_configs:
                logger.info(f"WenetSpeech download completed successfully")
                logger.info(f"Downloaded configurations: {sorted(downloaded_configs)}")
                
                # Clean up raw compressed files to save space
                raw_dir = dataset_dir / "raw"
                if raw_dir.exists():
                    import shutil
                    shutil.rmtree(raw_dir)
                    logger.info("Cleaned up raw compressed files")
                
                return True
            else:
                logger.error("Failed to download essential L_fixed configuration")
                return False
            
        except Exception as e:
            logger.error(f"Failed to download WenetSpeech: {e}")
            logger.info("Make sure you have requested access via the Google Form")
            return False
    
    def download_wenetspeech4tts(self) -> bool:
        """Download Wenetspeech4TTS/WenetSpeech4TTS dataset."""
        logger.info("Downloading WenetSpeech4TTS dataset...")
        dataset_dir = self.base_dir / "WenetSpeech4TTS"
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            from huggingface_hub import snapshot_download
            
            # Download the entire repository
            snapshot_download(
                repo_id="Wenetspeech4TTS/WenetSpeech4TTS",
                repo_type="dataset",
                local_dir=str(dataset_dir),
                token=self.hf_token
            )
            
            logger.info(f"WenetSpeech4TTS downloaded to {dataset_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download WenetSpeech4TTS: {e}")
            return False
    
    def download_emilia_dataset(self) -> bool:
        """Download amphion/Emilia-Dataset."""
        logger.info("Downloading Emilia Dataset...")
        dataset_dir = self.base_dir / "Emilia-Dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            from huggingface_hub import snapshot_download
            
            # Download the entire repository (this will be large!)
            logger.warning("Emilia Dataset is 4.5TB. This will take a very long time!")
            snapshot_download(
                repo_id="amphion/Emilia-Dataset",
                repo_type="dataset",
                local_dir=str(dataset_dir),
                token=self.hf_token
            )
            
            logger.info(f"Emilia Dataset downloaded to {dataset_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Emilia Dataset: {e}")
            return False
    
    def verify_download(self, dataset_name: str) -> bool:
        """Verify that a dataset was downloaded correctly."""
        dataset_dir = self.base_dir / dataset_name.split('/')[-1]
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory {dataset_dir} does not exist")
            return False
        
        # Check for metadata files
        metadata_files = list(dataset_dir.glob("*.jsonl"))
        if not metadata_files:
            logger.warning(f"No metadata files found in {dataset_dir}")
        
        # Special verification for WenetSpeech
        if "wenetspeech" in dataset_name.lower():
            l_fixed_file = dataset_dir / "L_fixed_metadata.jsonl"
            if l_fixed_file.exists():
                # Count lines to verify content
                with open(l_fixed_file, 'r') as f:
                    line_count = sum(1 for _ in f)
                logger.info(f"WenetSpeech L_fixed contains {line_count} samples")
                if line_count > 0:
                    logger.info(f"WenetSpeech verification successful")
                    
                    # Also check for other configurations
                    other_configs = ["DEV_fixed", "TEST_NET", "TEST_MEETING"]
                    for config in other_configs:
                        config_file = dataset_dir / f"{config}_metadata.jsonl"
                        if config_file.exists():
                            with open(config_file, 'r') as f:
                                config_lines = sum(1 for _ in f)
                            logger.info(f"WenetSpeech {config} contains {config_lines} samples")
                    
                    return True
            else:
                logger.error("WenetSpeech L_fixed file not found")
                return False
        
        # Check for audio directories for other datasets
        audio_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        if not audio_dirs and not metadata_files:
            logger.warning(f"No audio directories or metadata files found in {dataset_dir}")
        
        logger.info(f"Dataset {dataset_name} verification completed")
        return True
    
    def download_dataset(self, dataset_name: str) -> bool:
        """Download a specific dataset."""
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        config = self.datasets[dataset_name]
        
        if config['requires_approval'] and not self.hf_token:
            logger.error(f"Dataset {dataset_name} requires HuggingFace token")
            return False
        
        logger.info(f"Starting download of {dataset_name} (Priority: {config['priority']}, Size: {config['size_gb']}GB)")
        
        success = False
        if dataset_name == 'shenyunhang/AudioQA-1M':
            success = self.download_audioqa_1m()
        elif dataset_name == 'wenet-e2e/wenetspeech':
            success = self.download_wenetspeech()
        elif dataset_name == 'Wenetspeech4TTS/WenetSpeech4TTS':
            success = self.download_wenetspeech4tts()
        elif dataset_name == 'amphion/Emilia-Dataset':
            success = self.download_emilia_dataset()
        
        if success:
            self.verify_download(dataset_name)
            self.stats['downloaded'] += 1
            self.stats['total_size'] += config['size_gb']
        else:
            self.stats['errors'] += 1
        
        return success
    
    def download_all(self, priority_filter: Optional[int] = None) -> Dict[str, bool]:
        """Download all datasets, optionally filtered by priority."""
        results = {}
        
        # Sort datasets by priority
        sorted_datasets = sorted(
            self.datasets.items(),
            key=lambda x: x[1]['priority']
        )
        
        for dataset_name, config in sorted_datasets:
            if priority_filter and config['priority'] > priority_filter:
                logger.info(f"Skipping {dataset_name} (priority {config['priority']} > {priority_filter})")
                continue
            
            results[dataset_name] = self.download_dataset(dataset_name)
        
        return results
    
    def print_summary(self):
        """Print download summary."""
        logger.info("=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Datasets downloaded: {self.stats['downloaded']}")
        logger.info(f"Total size: {self.stats['total_size']:.1f} GB")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Available disk space: {self._get_available_space():.1f} GB")
        logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Download VITA-Audio datasets")
    parser.add_argument("--base-dir", default="./datasets", help="Base directory for datasets")
    parser.add_argument("--hf-token", help="HuggingFace access token")
    parser.add_argument("--dataset", help="Download specific dataset")
    parser.add_argument("--priority", type=int, help="Download only datasets with priority <= N")
    parser.add_argument("--check-only", action="store_true", help="Only check requirements")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DatasetDownloader(args.base_dir, args.hf_token)
    
    # Check requirements
    if not downloader.check_requirements():
        sys.exit(1)
    
    if args.check_only:
        logger.info("Requirements check passed")
        return
    
    # Setup HuggingFace if token provided
    if args.hf_token:
        downloader.setup_huggingface()
    
    # Download datasets
    if args.dataset:
        success = downloader.download_dataset(args.dataset)
        if not success:
            sys.exit(1)
    else:
        results = downloader.download_all(args.priority)
        failed = [name for name, success in results.items() if not success]
        if failed:
            logger.error(f"Failed to download: {failed}")
    
    # Print summary
    downloader.print_summary()

if __name__ == "__main__":
    main()
