"""
Driver program for stress testing based self-consistency.

Usage:
    python run.py --problem problem.txt --output ./output --candidates 9 --stress 5

Environment:
    OPENAI_API_KEY must be set
"""
import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime

from generate_candidates import generate_candidates
from generate_stress_candidates import generate_stress_candidates
from generator import generate_tests
from executor import (
    verify_sample_tests,
    build_stress_predicted_tests,
    filter_by_stress_tests,
    majority_vote
)


def load_problem_file(problem_path: str) -> dict:
    """
    Load problem from file. Expects JSON format:
    {
        "title": "Problem Title",
        "description": "Full problem description",
        "sample_tests": [
            {"input": "test input", "output": "expected output"},
            ...
        ],
        "additional_tests": [
            {"input": "test input only"},
            ...
        ]
    }
    """
    with open(problem_path, 'r') as f:
        return json.load(f)


async def run_pipeline(problem_data: dict, output_dir: Path, num_candidates: int = 9, num_stress: int = 5, model: str = "gpt-4o"):
    """
    Run the complete stress testing pipeline.
    
    Pipeline:
    1. Generate candidates and stress candidates in parallel
    2. Verify candidates against sample tests
    3. Build stress-predicted tests from stress candidates
    4. Filter candidates using stress tests
    5. Majority vote on remaining candidates
    """
    print("=" * 80)
    print("STRESS TESTING BASED SELF-CONSISTENCY PIPELINE")
    print("=" * 80)
    
    # Create output directories
    candidate_dir = output_dir / "candidates"
    stress_dir = output_dir / "stress"
    
    # Prepare problem text and test data
    problem_text = f"{problem_data['title']}\n\n{problem_data['description']}"
    sample_tests = [(t['input'], t['output']) for t in problem_data.get('sample_tests', [])]
    additional_inputs = [t['input'] for t in problem_data.get('additional_tests', [])]
    
    # Step 1: Generate candidates; also generate extra test inputs to use for stress testing
    print("\n[STEP 1] Generating candidates and extra test inputs for stress testing...")
    start_time = datetime.now()

    # generate extra test inputs (used by stress generation and later for stress-prediction)
    gen_dir = output_dir / "generated_tests"
    gen_task = generate_tests(problem_text, str(gen_dir), num=num_stress)

    # generate candidate solutions in parallel with test generation
    cand_task = generate_candidates(problem_text, str(candidate_dir), num_candidates, model)
    
    cand_count, gen_tests = await asyncio.gather(cand_task, gen_task)

    # write generated tests into additional_inputs so stress and executor see them
    generated_inputs = gen_tests or []
    if generated_inputs:
        additional_inputs.extend(generated_inputs)

    # now generate stress candidates (can use generated tests indirectly via prompt if desired)
    stress_task = generate_stress_candidates(problem_text, str(stress_dir), num_stress, model)
    stress_count = await stress_task
    
    gen_time = (datetime.now() - start_time).total_seconds()
    print(f"\n  Generation completed in {gen_time:.2f}s")
    print(f"  Candidates: {cand_count}, Stress: {stress_count}, Generated tests: {len(generated_inputs)}")
    
    if cand_count == 0:
        print("\n❌ No candidates generated. Exiting.")
        return None
    
    # Step 2: Verify candidates against sample tests
    print("\n[STEP 2] Verifying candidates against sample tests...")
    verified_candidates = await verify_sample_tests(candidate_dir, sample_tests, timeout=2.0)
    
    if not verified_candidates:
        print("\n❌ No candidates passed sample tests. Exiting.")
        return None
    
    # Step 3: Build stress-predicted tests
    print("\n[STEP 3] Building stress-predicted test suite...")
    stress_predicted = await build_stress_predicted_tests(stress_dir, additional_inputs, timeout=5.0, min_agree=2)
    
    # Step 4: Filter candidates using stress tests
    print("\n[STEP 4] Filtering candidates with stress tests...")
    filtered_candidates = await filter_by_stress_tests(verified_candidates, stress_predicted, additional_inputs, timeout=2.0)
    
    # Use verified candidates if no candidates pass stress filtering
    final_pool = filtered_candidates if filtered_candidates else verified_candidates
    
    # Step 5: Majority vote
    print("\n[STEP 5] Performing majority vote...")
    selected = await majority_vote(final_pool, additional_inputs, timeout=2.0)
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Candidates generated:        {cand_count}")
    print(f"Stress candidates generated: {stress_count}")
    print(f"Passed sample tests:         {len(verified_candidates)}")
    print(f"Stress-predicted tests:      {len(stress_predicted)}/{len(additional_inputs)}")
    print(f"Passed stress filtering:     {len(filtered_candidates)}")
    print(f"Final selected:              {selected.name if selected else 'None'}")
    print("=" * 80)
    
    if selected:
        print(f"\n✅ Final solution: {selected}")
        # Copy to final output
        final_output = output_dir / "final_solution.py"
        with open(selected, 'r') as src, open(final_output, 'w') as dst:
            dst.write(src.read())
        print(f"   Copied to: {final_output}")
    
    return selected


async def main():
    parser = argparse.ArgumentParser(description="Run stress testing based self-consistency")
    parser.add_argument("--problem", required=True, help="Path to problem JSON file")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--candidates", type=int, default=9, help="Number of candidates to generate")
    parser.add_argument("--stress", type=int, default=5, help="Number of stress candidates to generate")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    
    args = parser.parse_args()
    
    # Load problem
    print(f"Loading problem from: {args.problem}")
    problem_data = load_problem_file(args.problem)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    await run_pipeline(
        problem_data,
        output_dir,
        num_candidates=args.candidates,
        num_stress=args.stress,
        model=args.model
    )


if __name__ == "__main__":
    asyncio.run(main())
