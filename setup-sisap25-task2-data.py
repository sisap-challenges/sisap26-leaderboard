#!/usr/bin/env python3
"""
SISAP25 Challenge Task 2 Data Setup Script

This script automatically downloads and sets up the data directory for Task 2 
of the SISAP25 Indexing Challenge (k-NN graph construction on GOOAQ dataset).

Task 2 specifications:
- Dataset: GOOAQ (3M vectors, 384 dimensions)
- Objective: k=15 nearest neighbor graph construction
- k-value: 15 nearest neighbors per point
- Similarity metric: Cosine similarity / dot product
- Resource limits: 8 CPUs, 16GB RAM, 12-hour time limit
- Minimum recall: 0.8 for ranking

Usage:
    python setup-sisap25-task2-data.py [--data-dir DATA_DIR] [--include-eval] [--verify-only]
    
Requirements:
    pip install requests h5py tqdm
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

try:
    import h5py
    import requests
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required dependency - {e}")
    print("Please install required packages: pip install requests h5py tqdm")
    sys.exit(1)


class SISAP25DataSetup:
    """SISAP25 Task 2 data setup and validation utility."""
    
    # HuggingFace dataset repository base URL
    BASE_URL = "https://huggingface.co/datasets/sadit/SISAP2025/resolve/main"
    
    # Required files for Task 2
    TASK2_FILES = {
        "allknn-benchmark-dev-gooaq.h5": {
            "size_mb": 768,
            "description": "Task 2 specific gold standard (k=15 nearest neighbors)",
            "required": True
        },
        "benchmark-dev-gooaq.h5": {
            "size_mb": 4820, 
            "description": "Full development dataset with train/test splits",
            "required": True
        },
        "benchmark-eval-gooaq.h5": {
            "size_mb": 7880,
            "description": "Final evaluation dataset (larger, for submission)",
            "required": False
        }
    }
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data setup manager.
        
        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, filename: str, show_progress: bool = True) -> bool:
        """Download a single file from HuggingFace repository.
        
        Args:
            filename: Name of the file to download
            show_progress: Whether to show download progress bar
            
        Returns:
            True if download successful, False otherwise
        """
        url = f"{self.BASE_URL}/{filename}"
        filepath = self.data_dir / filename
        
        # Check if file already exists
        if filepath.exists():
            print(f"✓ {filename} already exists, skipping download")
            return True
            
        print(f"📥 Downloading {filename}...")
        print(f"    URL: {url}")
        print(f"    Size: ~{self.TASK2_FILES[filename]['size_mb']} MB")
        
        try:
            # Get file size for progress bar
            response = requests.head(url, allow_redirects=True)
            if response.status_code != 200:
                print(f"❌ Error: File not found at {url} (HTTP {response.status_code})")
                return False
                
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                if show_progress and total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
            print(f"✅ Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            # Clean up partial download
            if filepath.exists():
                filepath.unlink()
            return False
    
    def verify_hdf5_file(self, filename: str) -> Tuple[bool, dict]:
        """Verify HDF5 file structure and extract metadata.
        
        Args:
            filename: Name of the HDF5 file to verify
            
        Returns:
            Tuple of (is_valid, metadata_dict)
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            return False, {"error": "File does not exist"}
            
        try:
            with h5py.File(filepath, 'r') as f:
                metadata = {
                    "file_size_mb": filepath.stat().st_size / (1024 * 1024),
                    "keys": list(f.keys()),
                    "attributes": dict(f.attrs),
                }
                
                # Task 2 specific validation
                if filename == "allknn-benchmark-dev-gooaq.h5":
                    # Should contain knns and dists arrays
                    if 'knns' in f and 'dists' in f:
                        metadata["knns_shape"] = f['knns'].shape
                        metadata["dists_shape"] = f['dists'].shape
                        metadata["task2_format"] = True
                    else:
                        metadata["task2_format"] = False
                        metadata["error"] = "Missing required arrays for Task 2"
                        
                elif filename.startswith("benchmark-"):
                    # Should contain train data and test sets
                    if 'train' in f:
                        metadata["train_shape"] = f['train'].shape
                        metadata["dimensions"] = f['train'].shape[0] if len(f['train'].shape) > 1 else "N/A"
                        metadata["num_vectors"] = f['train'].shape[1] if len(f['train'].shape) > 1 else f['train'].shape[0]
                        
                    # Check for test sets
                    test_sets = []
                    for key in ['itest', 'otest']:
                        if key in f:
                            test_sets.append(key)
                            if 'queries' in f[key]:
                                metadata[f"{key}_queries_shape"] = f[key]['queries'].shape
                                
                    metadata["test_sets"] = test_sets
                    
            return True, metadata
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def setup_data_directory(self, include_eval: bool = False) -> bool:
        """Set up complete data directory for Task 2.
        
        Args:
            include_eval: Whether to download evaluation dataset (7.8GB)
            
        Returns:
            True if setup successful, False otherwise
        """
        print("🚀 Setting up SISAP25 Task 2 data directory...")
        print(f"📁 Data directory: {self.data_dir.absolute()}")
        print()
        
        # Download required files
        success = True
        for filename, info in self.TASK2_FILES.items():
            if not info["required"] and not include_eval:
                if filename == "benchmark-eval-gooaq.h5":
                    print(f"⏭️  Skipping {filename} (evaluation dataset, use --include-eval to download)")
                    continue
                    
            print(f"📋 {info['description']}")
            if not self.download_file(filename):
                success = False
                
        print()
        return success
    
    def verify_setup(self) -> bool:
        """Verify downloaded files and show dataset information.
        
        Returns:
            True if all files are valid, False otherwise
        """
        print("🔍 Verifying downloaded datasets...")
        print()
        
        all_valid = True
        
        for filename in self.TASK2_FILES.keys():
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                if self.TASK2_FILES[filename]["required"]:
                    print(f"❌ {filename}: Missing (required)")
                    all_valid = False
                else:
                    print(f"⏭️  {filename}: Not downloaded (optional)")
                continue
                
            is_valid, metadata = self.verify_hdf5_file(filename)
            
            if is_valid:
                print(f"✅ {filename}: Valid")
                print(f"    Size: {metadata['file_size_mb']:.1f} MB")
                print(f"    Keys: {metadata['keys']}")
                
                # Show specific info for Task 2 gold standard
                if filename == "allknn-benchmark-dev-gooaq.h5":
                    if metadata.get("task2_format"):
                        print(f"    k-NN matrix: {metadata['knns_shape']}")
                        print(f"    Distance matrix: {metadata['dists_shape']}")
                    else:
                        print(f"    ⚠️  Warning: {metadata.get('error', 'Invalid format')}")
                        
                # Show info for benchmark datasets
                elif filename.startswith("benchmark-"):
                    if 'train_shape' in metadata:
                        print(f"    Training data: {metadata['train_shape']} ({metadata['dimensions']}D vectors)")
                        print(f"    Number of vectors: {metadata['num_vectors']:,}")
                    if metadata.get('test_sets'):
                        print(f"    Test sets: {', '.join(metadata['test_sets'])}")
                        
            else:
                print(f"❌ {filename}: Invalid - {metadata.get('error', 'Unknown error')}")
                all_valid = False
                
            print()
            
        return all_valid
    
    def show_usage_info(self):
        """Display information about using the setup data."""
        print("📖 Task 2 Dataset Usage Information")
        print("=" * 50)
        print()
        print("🎯 Task 2 Objective:")
        print("   Construct k=15 nearest neighbor graph for GOOAQ dataset")
        print("   - 3,012,496 vectors (3M)")
        print("   - 384 dimensions")
        print("   - Cosine similarity / dot product")
        print("   - Minimum 0.8 recall required for ranking")
        print()
        print("📁 File Usage:")
        print("   • allknn-benchmark-dev-gooaq.h5: Gold standard for Task 2 evaluation")
        print("   • benchmark-dev-gooaq.h5: Main dataset with train/test splits")
        print("   • benchmark-eval-gooaq.h5: Final evaluation dataset (optional)")
        print()
        print("🐍 Python Example:")
        print("   import h5py")
        print("   with h5py.File('data/benchmark-dev-gooaq.h5', 'r') as f:")
        print("       train_data = f['train'][:]  # Shape: (384, 3012496)")
        print("       print(f'Dataset shape: {train_data.shape}')")
        print()
        print("   with h5py.File('data/allknn-benchmark-dev-gooaq.h5', 'r') as f:")
        print("       gold_knns = f['knns'][:]    # Gold standard k-NN (15, n)")
        print("       gold_dists = f['dists'][:]  # Corresponding distances")
        print()
        print("⚡ Resource Constraints:")
        print("   • Container: 8 CPUs, 16GB RAM")
        print("   • Time limit: 12 hours") 
        print("   • Algorithm: Your k-NN graph construction implementation")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Setup data directory for SISAP25 Challenge Task 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic setup with required files
  python setup-sisap25-task2-data.py
  
  # Include evaluation dataset (7.8GB additional)
  python setup-sisap25-task2-data.py --include-eval
  
  # Custom data directory
  python setup-sisap25-task2-data.py --data-dir /path/to/sisap25/data
  
  # Only verify existing files
  python setup-sisap25-task2-data.py --verify-only
        """)
    
    parser.add_argument(
        "--data-dir", 
        default="data",
        help="Directory to store datasets (default: ./data)"
    )
    parser.add_argument(
        "--include-eval",
        action="store_true", 
        help="Download evaluation dataset (additional 7.8GB)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing files, don't download"
    )
    
    args = parser.parse_args()
    
    # Initialize setup manager
    setup = SISAP25DataSetup(args.data_dir)
    
    # Verify only mode
    if args.verify_only:
        if setup.verify_setup():
            print("✅ All datasets are valid!")
            setup.show_usage_info()
            return 0
        else:
            print("❌ Some datasets are missing or invalid")
            return 1
    
    # Download and setup
    print("SISAP25 Challenge - Task 2 Data Setup")
    print("=====================================")
    print()
    
    if setup.setup_data_directory(args.include_eval):
        print("✅ Data setup completed successfully!")
        print()
        
        if setup.verify_setup():
            print("✅ All datasets verified!")
            setup.show_usage_info()
            return 0
        else:
            print("❌ Verification failed")
            return 1
    else:
        print("❌ Data setup failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())