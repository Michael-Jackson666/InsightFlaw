"""
SIFT1B Dataset Download Script
==============================
Downloads the SIFT1B dataset from INRIA BigANN project.

Usage:
    python download.py              # Download all files
    python download.py --query-only # Download query vectors only (for testing)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from config import DATA_DIR, DOWNLOAD_URLS


def check_disk_space(required_gb: float) -> bool:
    """Check if sufficient disk space is available."""
    import shutil
    
    total, used, free = shutil.disk_usage(DATA_DIR.parent)
    free_gb = free / (1024 ** 3)
    
    print(f"💾 Disk space: {free_gb:.1f} GB free")
    
    if free_gb < required_gb:
        print(f"⚠️  Warning: {required_gb:.1f} GB required, only {free_gb:.1f} GB available")
        return False
    return True


def download_file(url: str, output_path: Path, description: str) -> bool:
    """Download a file using wget."""
    print(f"\n📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Output: {output_path}")
    
    if output_path.exists():
        print(f"   ⏭️  File already exists, skipping")
        return True
    
    # Create parent directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use wget for FTP download with progress
        cmd = ["wget", "-c", "--progress=bar:force", "-O", str(output_path), url]
        result = subprocess.run(cmd, check=True)
        print(f"   ✅ Download complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Download failed: {e}")
        return False
    except FileNotFoundError:
        print("   ❌ wget not found. Please install wget:")
        print("      brew install wget  # macOS")
        return False


def decompress_file(filepath: Path) -> bool:
    """Decompress .gz or .tar.gz files."""
    if not filepath.exists():
        print(f"   ❌ File not found: {filepath}")
        return False
    
    output_path = filepath.with_suffix("") if filepath.suffix == ".gz" else filepath.parent
    
    if filepath.suffix == ".gz" and not str(filepath).endswith(".tar.gz"):
        # .bvecs.gz file
        if output_path.exists():
            print(f"   ⏭️  Already decompressed: {output_path.name}")
            return True
        
        print(f"   📦 Decompressing {filepath.name}...")
        try:
            subprocess.run(["gunzip", "-k", str(filepath)], check=True)
            print(f"   ✅ Decompressed to {output_path.name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Decompression failed: {e}")
            return False
    
    elif str(filepath).endswith(".tar.gz"):
        # .tar.gz file
        print(f"   📦 Extracting {filepath.name}...")
        try:
            subprocess.run(["tar", "-xzf", str(filepath), "-C", str(filepath.parent)], check=True)
            print(f"   ✅ Extracted")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Extraction failed: {e}")
            return False
    
    return True


def download_query_only():
    """Download only query vectors and ground truth (for testing)."""
    print("=" * 60)
    print("📥 Downloading Query Vectors and Ground Truth")
    print("=" * 60)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download query vectors (~1 MB)
    query_gz = DATA_DIR / "bigann_query.bvecs.gz"
    if download_file(DOWNLOAD_URLS["query"], query_gz, "query vectors"):
        decompress_file(query_gz)
    
    # Download ground truth (~40 MB)
    gnd_gz = DATA_DIR / "bigann_gnd.tar.gz"
    if download_file(DOWNLOAD_URLS["groundtruth"], gnd_gz, "ground truth"):
        decompress_file(gnd_gz)
    
    print("\n✅ Query data download complete!")


def download_all():
    """Download complete SIFT1B dataset."""
    print("=" * 60)
    print("📥 Downloading SIFT1B Dataset (1 Billion Vectors)")
    print("=" * 60)
    print("\n⚠️  WARNING: This will download ~130 GB of data!")
    print("   Estimated time: Several hours depending on connection\n")
    
    # Check disk space (need ~300 GB for download + decompression)
    if not check_disk_space(300):
        response = input("\nContinue anyway? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download base vectors (~128 GB compressed)
    base_gz = DATA_DIR / "bigann_base.bvecs.gz"
    if download_file(DOWNLOAD_URLS["base"], base_gz, "base vectors (128 GB)"):
        decompress_file(base_gz)
    
    # Download query vectors
    query_gz = DATA_DIR / "bigann_query.bvecs.gz"
    if download_file(DOWNLOAD_URLS["query"], query_gz, "query vectors"):
        decompress_file(query_gz)
    
    # Download ground truth
    gnd_gz = DATA_DIR / "bigann_gnd.tar.gz"
    if download_file(DOWNLOAD_URLS["groundtruth"], gnd_gz, "ground truth"):
        decompress_file(gnd_gz)
    
    print("\n" + "=" * 60)
    print("✅ SIFT1B Dataset Download Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download SIFT1B dataset for billion-scale vector search benchmark"
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Download only query vectors and ground truth (for testing pipeline)",
    )
    parser.add_argument(
        "--check-space",
        action="store_true",
        help="Check available disk space and exit",
    )
    
    args = parser.parse_args()
    
    if args.check_space:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        check_disk_space(300)
        return
    
    if args.query_only:
        download_query_only()
    else:
        download_all()


if __name__ == "__main__":
    main()
