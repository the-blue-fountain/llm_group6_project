"""
Execute candidate solutions and apply stress filtering.
"""
import asyncio
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict


class CodeExecutor:
    """Execute Python code files asynchronously with timeouts."""
    
    def __init__(self, max_concurrency: int = 20):
        self._sem = asyncio.Semaphore(max_concurrency)
    
    async def run_file(self, code_path: Path, input_data: str, timeout: float) -> Tuple[bool, str]:
        """
        Execute a Python file with given input.
        
        Returns:
            (success, output) tuple
        """
        async with self._sem:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "python3",
                    str(code_path),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=input_data.encode()),
                    timeout=timeout
                )
                
                if proc.returncode != 0:
                    return False, f"ERROR: {stderr.decode().strip()}"
                
                return True, stdout.decode().strip()
                
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except:
                    pass
                return False, "TIMEOUT"
            except Exception as e:
                return False, f"EXCEPTION: {str(e)}"
    
    async def run_file_on_tests(self, code_path: Path, test_inputs: List[str], timeout: float) -> List[Tuple[bool, str]]:
        """Run a single code file on multiple test inputs in parallel."""
        tasks = [self.run_file(code_path, inp, timeout) for inp in test_inputs]
        return await asyncio.gather(*tasks)


async def verify_sample_tests(candidate_dir: Path, sample_tests: List[Tuple[str, str]], timeout: float = 2.0) -> List[Path]:
    """
    Verify candidate solutions against sample tests.
    
    Returns:
        List of candidate files that pass all sample tests
    """
    executor = CodeExecutor(max_concurrency=30)
    candidate_files = sorted(candidate_dir.glob("candidate_*.py"))
    
    print(f"\nVerifying {len(candidate_files)} candidates against {len(sample_tests)} sample tests...")
    
    async def check_candidate(cand_path: Path) -> Optional[Path]:
        for inp, expected in sample_tests:
            success, output = await executor.run_file(cand_path, inp, timeout)
            if not success or output.strip() != expected.strip():
                return None
        return cand_path
    
    results = await asyncio.gather(*[check_candidate(c) for c in candidate_files])
    passing = [r for r in results if r is not None]
    
    print(f"  {len(passing)}/{len(candidate_files)} candidates passed sample tests")
    return passing


async def build_stress_predicted_tests(stress_dir: Path, additional_inputs: List[str], timeout: float = 5.0, min_agree: int = 2) -> Dict[int, str]:
    """
    Run stress candidates on additional test inputs and build predicted outputs.
    
    Returns:
        Dict mapping test_index -> predicted_output for reliable tests
    """
    executor = CodeExecutor(max_concurrency=20)
    stress_files = sorted(stress_dir.glob("stress_*.py"))
    
    print(f"\nRunning {len(stress_files)} stress candidates on {len(additional_inputs)} additional tests...")
    
    # Collect outputs from all stress candidates
    results_by_test: Dict[int, List[str]] = defaultdict(list)
    
    async def run_stress_candidate(stress_path: Path):
        results = await executor.run_file_on_tests(stress_path, additional_inputs, timeout)
        for idx, (success, output) in enumerate(results):
            if success:
                results_by_test[idx].append(output)
    
    await asyncio.gather(*[run_stress_candidate(s) for s in stress_files])
    
    # Build predicted outputs using majority vote (mode with count >= min_agree)
    stress_predicted: Dict[int, str] = {}
    for idx, outputs in results_by_test.items():
        if not outputs:
            continue
        counter = Counter(outputs)
        most_common_output, count = counter.most_common(1)[0]
        if count >= min_agree:
            stress_predicted[idx] = most_common_output
    
    print(f"  Built {len(stress_predicted)}/{len(additional_inputs)} reliable stress-predicted tests")
    return stress_predicted


async def filter_by_stress_tests(candidate_files: List[Path], stress_predicted: Dict[int, str], additional_inputs: List[str], timeout: float = 2.0) -> List[Path]:
    """
    Filter candidates by running them on stress-predicted tests.
    
    Returns:
        List of candidate files that pass all stress-predicted tests
    """
    if not stress_predicted:
        print("\nNo stress-predicted tests available, skipping stress filtering")
        return candidate_files
    
    executor = CodeExecutor(max_concurrency=30)
    
    print(f"\nFiltering {len(candidate_files)} candidates with {len(stress_predicted)} stress tests...")
    
    async def check_candidate(cand_path: Path) -> Optional[Path]:
        # Run only on stress-predicted test indices
        test_indices = sorted(stress_predicted.keys())
        test_inputs_subset = [additional_inputs[i] for i in test_indices]
        
        results = await executor.run_file_on_tests(cand_path, test_inputs_subset, timeout)
        
        for (success, output), idx in zip(results, test_indices):
            expected = stress_predicted[idx]
            if not success or output.strip() != expected.strip():
                return None
        return cand_path
    
    results = await asyncio.gather(*[check_candidate(c) for c in candidate_files])
    passing = [r for r in results if r is not None]
    
    print(f"  {len(passing)}/{len(candidate_files)} candidates passed stress filtering")
    return passing


async def majority_vote(candidate_files: List[Path], additional_inputs: List[str], timeout: float = 2.0) -> Optional[Path]:
    """
    Perform majority voting on candidates using all additional test inputs.
    
    Returns:
        Path to the selected candidate
    """
    if not candidate_files:
        return None
    
    executor = CodeExecutor(max_concurrency=30)
    
    print(f"\nMajority voting among {len(candidate_files)} candidates on {len(additional_inputs)} tests...")
    
    # Run all candidates on all additional tests
    async def get_outputs(cand_path: Path) -> Tuple[Path, Tuple[str, ...]]:
        results = await executor.run_file_on_tests(cand_path, additional_inputs, timeout)
        outputs = tuple(output if success else "__ERROR__" for success, output in results)
        return cand_path, outputs
    
    results = await asyncio.gather(*[get_outputs(c) for c in candidate_files])
    
    # Cluster by output signature
    clusters: Dict[Tuple[str, ...], List[Path]] = defaultdict(list)
    for cand_path, outputs in results:
        clusters[outputs].append(cand_path)
    
    # Select from largest cluster
    largest_cluster = max(clusters.values(), key=len)
    selected = largest_cluster[0]
    
    print(f"  Largest cluster size: {len(largest_cluster)}/{len(candidate_files)}")
    print(f"  Selected: {selected.name}")
    
    return selected
