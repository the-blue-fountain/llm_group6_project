"""
Extract USACO dataset from parquet file.
Downloads test data from links and organizes into problem directories.

Usage:
    python extract_usaco.py --parquet <path_or_url> --output usaco/
"""
import os
import sys
import argparse
import json
import requests
import zipfile
import io
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def download_file(url: str, retries: int = 3) -> bytes:
    """Download file from URL with retries."""
    for attempt in range(retries):
        try:
            print(f"  Downloading: {url} (attempt {attempt + 1}/{retries})")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"  Error downloading {url}: {e}")
            if attempt == retries - 1:
                raise
    return b""


def extract_test_data(test_data_link: str, output_dir: Path) -> int:
    """
    Download and extract test data from link.
    Returns number of test cases extracted.
    """
    if not test_data_link or test_data_link == "N/A":
        return 0
    
    try:
        # Download test data (usually a zip file)
        content = download_file(test_data_link)
        
        # Create input/output directories
        input_dir = output_dir / "golden" / "inputs"
        output_dir_path = output_dir / "golden" / "outputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Extract zip file
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for member in zf.namelist():
                # Typically .in files are inputs and .out files are outputs
                if member.endswith('.in'):
                    test_name = Path(member).stem
                    data = zf.read(member)
                    with open(input_dir / f"{test_name}.txt", "wb") as f:
                        f.write(data)
                elif member.endswith('.out'):
                    test_name = Path(member).stem
                    data = zf.read(member)
                    with open(output_dir_path / f"{test_name}.txt", "wb") as f:
                        f.write(data)
        
        # Count test cases
        test_count = len(list(input_dir.glob("*.txt")))
        return test_count
        
    except Exception as e:
        print(f"  Warning: Failed to extract test data from {test_data_link}: {e}")
        return 0


def create_problem_directory(problem: Dict[str, Any], output_base: Path) -> bool:
    """
    Create directory structure for a single problem.
    Returns True if successful.
    """
    try:
        # Create problem directory using name
        problem_name = problem.get('name', '').replace('/', '_').replace(' ', '_')
        if not problem_name:
            print("  Warning: Problem missing name, skipping")
            return False
        
        problem_dir = output_base / problem_name
        problem_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {problem_name}")
        
        # Parse samples into structured format
        samples = []
        samples_raw = problem.get('samples', [])
        if isinstance(samples_raw, list):
            for idx, sample in enumerate(samples_raw):
                if isinstance(sample, dict):
                    samples.append({
                        "input": sample.get('input', ''),
                        "output": sample.get('output', '')
                    })
        
        # Create problem.json
        problem_data = {
            "title": problem.get('name', 'Untitled'),
            "description": problem.get('description', ''),
            "input_format": problem.get('input_format', ''),
            "output_format": problem.get('output_format', ''),
            "problem_level": problem.get('problem_level', 'unknown'),
            "problem_link": problem.get('problem_link', ''),
            "solution_link": problem.get('solution_link', ''),
            "runtime_limit": problem.get('runtime_limit', 2.0),
            "memory_limit": problem.get('memory_limit', 256),
            "sample_tests": samples,
            "additional_tests": []  # Will be populated by generator.py during run
        }
        
        with open(problem_dir / "problem.json", "w") as f:
            json.dump(problem_data, f, indent=2)
        
        # Download and extract test data
        test_data_link = problem.get('test_data_link', '')
        test_count = extract_test_data(test_data_link, problem_dir)
        print(f"  Extracted {test_count} test cases")
        
        # Save official solution for reference (optional)
        if problem.get('solution'):
            solution_dir = problem_dir / "reference"
            solution_dir.mkdir(exist_ok=True)
            with open(solution_dir / "official_solution.txt", "w") as f:
                f.write(problem.get('solution', ''))
        
        print(f"  ✓ Created: {problem_dir}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing problem {problem.get('name', 'unknown')}: {e}")
        return False


def extract_usaco_dataset(parquet_path: str, output_dir: str):
    """
    Main extraction function.
    Reads parquet file and creates problem directories.
    """
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading parquet file: {parquet_path}")
    
    # Download parquet if it's a URL
    if parquet_path.startswith('http://') or parquet_path.startswith('https://'):
        print("Downloading parquet file...")
        content = download_file(parquet_path)
        df = pd.read_parquet(io.BytesIO(content))
    else:
        df = pd.read_parquet(parquet_path)
    
    print(f"Found {len(df)} problems in dataset")
    
    # Process each problem
    success_count = 0
    for idx, row in df.iterrows():
        problem = row.to_dict()
        if create_problem_directory(problem, output_base):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Extraction complete: {success_count}/{len(df)} problems successfully processed")
    print(f"Output directory: {output_base.absolute()}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Extract USACO dataset from parquet")
    parser.add_argument("--parquet", required=True, help="Path or URL to parquet file")
    parser.add_argument("--output", default="usaco", help="Output directory (default: usaco)")
    
    args = parser.parse_args()
    
    try:
        extract_usaco_dataset(args.parquet, args.output)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
