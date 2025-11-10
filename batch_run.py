"""
Batch runner for USACO problems.
Copies pipeline files into each problem workspace, runs stress testing pipeline,
validates against golden test data, and generates results report.

Usage:
    python batch_run.py --problems usaco/ --concurrency 4 --model o3-mini
"""
import asyncio
import argparse
import json
import shutil
import subprocess
import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict


# Files to copy into each problem workspace
PIPELINE_FILES = [
    "llm_client.py",
    "generate_candidates.py",
    "generate_stress_candidates.py",
    "generator.py",
    "executor.py",
    "run.py",
    "prompts/",
    ".env"
]


class ProblemRunner:
    """Run stress testing pipeline for a single problem."""
    
    def __init__(self, problem_dir: Path, model: str, candidates: int, stress: int, reasoning_effort: str):
        self.problem_dir = problem_dir
        self.model = model
        self.candidates = candidates
        self.stress = stress
        self.reasoning_effort = reasoning_effort
        self.workspace_dir = problem_dir / "workspace"
        self.results = {
            "problem_name": problem_dir.name,
            "problem_dir": str(problem_dir),
            "status": "pending",
            "error": None,
            "pipeline_time": 0,
            "validation_time": 0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "pass_rate": 0.0,
            "candidates_generated": 0,
            "stress_generated": 0,
            "gen_tests": 0
        }
    
    def setup_workspace(self):
        """Copy pipeline files into problem workspace."""
        self.workspace_dir.mkdir(exist_ok=True)
        
        for file_or_dir in PIPELINE_FILES:
            src = Path(file_or_dir)
            if not src.exists():
                continue
            
            dst = self.workspace_dir / src.name
            
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    
    def cleanup_workspace(self):
        """Remove workspace directory to restore clean state."""
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
    
    async def run_pipeline(self) -> bool:
        """Execute run.py in the problem workspace."""
        try:
            problem_json = self.problem_dir / "problem.json"
            if not problem_json.exists():
                self.results["error"] = "problem.json not found"
                self.results["status"] = "error"
                return False
            
            print(f"\n[{self.problem_dir.name}] Running stress testing pipeline...")
            start_time = time.time()
            
            # Run the pipeline
            cmd = [
                "python3",
                "run.py",
                "--problem", str(problem_json.absolute()),
                "--output", "./output",
                "--candidates", str(self.candidates),
                "--stress", str(self.stress),
                "--model", self.model
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.workspace_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            self.results["pipeline_time"] = time.time() - start_time
            
            if process.returncode != 0:
                self.results["error"] = f"Pipeline failed: {stderr.decode()}"
                self.results["status"] = "pipeline_failed"
                print(f"[{self.problem_dir.name}] ✗ Pipeline failed")
                return False
            
            # Parse output for metrics
            output_text = stdout.decode()
            self._parse_pipeline_output(output_text)
            
            print(f"[{self.problem_dir.name}] ✓ Pipeline completed in {self.results['pipeline_time']:.2f}s")
            return True
            
        except Exception as e:
            self.results["error"] = f"Pipeline error: {str(e)}"
            self.results["status"] = "pipeline_error"
            print(f"[{self.problem_dir.name}] ✗ Pipeline error: {e}")
            return False
    
    def _parse_pipeline_output(self, output: str):
        """Extract metrics from pipeline output."""
        for line in output.split('\n'):
            if "Candidates generated:" in line:
                try:
                    self.results["candidates_generated"] = int(line.split(':')[1].strip())
                except:
                    pass
            elif "Stress candidates generated:" in line:
                try:
                    self.results["stress_generated"] = int(line.split(':')[1].strip())
                except:
                    pass
            elif "Generated tests:" in line:
                try:
                    self.results["gen_tests"] = int(line.split(':')[1].strip())
                except:
                    pass
    
    async def validate_solution(self) -> bool:
        """Run final solution against golden test data."""
        try:
            final_solution = self.workspace_dir / "output" / "final_solution.py"
            if not final_solution.exists():
                self.results["error"] = "final_solution.py not found"
                self.results["status"] = "no_solution"
                return False
            
            golden_inputs = sorted((self.problem_dir / "golden" / "inputs").glob("*.txt"))
            golden_outputs = sorted((self.problem_dir / "golden" / "outputs").glob("*.txt"))
            
            if not golden_inputs:
                self.results["error"] = "No golden test data found"
                self.results["status"] = "no_tests"
                return False
            
            print(f"[{self.problem_dir.name}] Validating against {len(golden_inputs)} golden tests...")
            start_time = time.time()
            
            passed = 0
            total = 0
            
            for inp_file, out_file in zip(golden_inputs, golden_outputs):
                total += 1
                
                # Read input and expected output
                with open(inp_file, 'r') as f:
                    test_input = f.read()
                with open(out_file, 'r') as f:
                    expected_output = f.read().strip()
                
                # Run solution
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "python3",
                        str(final_solution.absolute()),
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=test_input.encode()),
                        timeout=10.0
                    )
                    
                    if proc.returncode == 0:
                        actual_output = stdout.decode().strip()
                        if actual_output == expected_output:
                            passed += 1
                    
                except asyncio.TimeoutError:
                    pass  # Count as failed
                except Exception:
                    pass  # Count as failed
            
            self.results["validation_time"] = time.time() - start_time
            self.results["total_tests"] = total
            self.results["passed_tests"] = passed
            self.results["failed_tests"] = total - passed
            self.results["pass_rate"] = (passed / total * 100) if total > 0 else 0.0
            self.results["status"] = "completed"
            
            print(f"[{self.problem_dir.name}] ✓ Validation: {passed}/{total} passed ({self.results['pass_rate']:.1f}%)")
            return True
            
        except Exception as e:
            self.results["error"] = f"Validation error: {str(e)}"
            self.results["status"] = "validation_error"
            print(f"[{self.problem_dir.name}] ✗ Validation error: {e}")
            return False
    
    async def run(self) -> Dict[str, Any]:
        """Run complete pipeline: setup -> run -> validate -> cleanup."""
        try:
            self.setup_workspace()
            
            if await self.run_pipeline():
                await self.validate_solution()
            
            self.cleanup_workspace()
            
        except Exception as e:
            self.results["error"] = f"Runner error: {str(e)}"
            self.results["status"] = "error"
            try:
                self.cleanup_workspace()
            except:
                pass
        
        return self.results


async def run_problem_batch(problem_dirs: List[Path], model: str, candidates: int, stress: int, reasoning_effort: str, concurrency: int) -> List[Dict[str, Any]]:
    """Run multiple problems with limited concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def run_with_semaphore(problem_dir: Path):
        async with semaphore:
            runner = ProblemRunner(problem_dir, model, candidates, stress, reasoning_effort)
            return await runner.run()
    
    tasks = [run_with_semaphore(p) for p in problem_dirs]
    return await asyncio.gather(*tasks)


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to CSV and JSON."""
    # Save CSV
    csv_file = output_file + ".csv"
    with open(csv_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\n✓ Results saved to {csv_file}")
    
    # Save JSON
    json_file = output_file + ".json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {json_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH RUN SUMMARY")
    print(f"{'='*80}")
    
    total = len(results)
    completed = sum(1 for r in results if r['status'] == 'completed')
    total_tests = sum(r['total_tests'] for r in results)
    total_passed = sum(r['passed_tests'] for r in results)
    
    print(f"Problems processed:  {total}")
    print(f"Successfully ran:    {completed} ({completed/total*100:.1f}%)")
    print(f"Total golden tests:  {total_tests}")
    print(f"Tests passed:        {total_passed} ({total_passed/total_tests*100:.1f}%)")
    
    # Breakdown by status
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r['status']] += 1
    
    print(f"\nStatus breakdown:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:20s}: {count}")
    
    print(f"{'='*80}")


async def main():
    parser = argparse.ArgumentParser(description="Batch run USACO problems")
    parser.add_argument("--problems", default="usaco", help="Directory containing problem folders")
    parser.add_argument("--filter", help="Filter problems by name pattern (regex)")
    parser.add_argument("--level", choices=["bronze", "silver", "gold", "platinum"], help="Filter by difficulty level")
    parser.add_argument("--limit", type=int, help="Limit number of problems to run")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of problems to run in parallel")
    parser.add_argument("--candidates", type=int, default=9, help="Number of candidate solutions per problem")
    parser.add_argument("--stress", type=int, default=5, help="Number of stress candidates per problem")
    parser.add_argument("--model", default="o3-mini", help="OpenAI model to use")
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"], help="Reasoning effort for o1/o3 models")
    parser.add_argument("--output", default="results", help="Output file prefix for results")
    
    args = parser.parse_args()
    
    # Find problem directories
    problems_base = Path(args.problems)
    if not problems_base.exists():
        print(f"Error: Problems directory not found: {problems_base}")
        return
    
    problem_dirs = [d for d in problems_base.iterdir() if d.is_dir() and (d / "problem.json").exists()]
    
    # Apply filters
    if args.filter:
        import re
        pattern = re.compile(args.filter)
        problem_dirs = [d for d in problem_dirs if pattern.search(d.name)]
    
    if args.level:
        filtered = []
        for d in problem_dirs:
            with open(d / "problem.json") as f:
                data = json.load(f)
                if data.get("problem_level", "").lower() == args.level.lower():
                    filtered.append(d)
        problem_dirs = filtered
    
    if args.limit:
        problem_dirs = problem_dirs[:args.limit]
    
    if not problem_dirs:
        print("No problems found matching criteria")
        return
    
    print(f"{'='*80}")
    print(f"BATCH RUN: {len(problem_dirs)} problems")
    print(f"Model: {args.model}")
    print(f"Candidates: {args.candidates}, Stress: {args.stress}")
    print(f"Concurrency: {args.concurrency}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Run all problems
    results = await run_problem_batch(
        problem_dirs,
        args.model,
        args.candidates,
        args.stress,
        args.reasoning_effort,
        args.concurrency
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal batch time: {total_time:.2f}s")
    
    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
