"""
Generate stress test candidates (brute-force solutions) for a problem.
Creates multiple brute-force solution files in the specified output directory.
"""
import asyncio
import os
from pathlib import Path
from llm_client import OpenAIClient


def load_prompt_template(template_name: str) -> str:
    """Load prompt template from prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / f"{template_name}.prompt"
    with open(prompt_path, "r") as f:
        return f.read()


def extract_code(response: str) -> str:
    """Extract Python code from LLM response, removing markdown if present."""
    lines = response.strip().split("\n")
    
    # Remove markdown code blocks if present
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    
    return "\n".join(lines)


async def generate_stress_candidates(problem: str, output_dir: str, num_stress: int = 5, model: str = "gpt-4o"):
    """
    Generate multiple brute-force stress test candidates in parallel.
    
    Args:
        problem: Problem description
        output_dir: Directory to save stress candidate files
        num_stress: Number of stress candidates to generate
        model: OpenAI model to use
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prompt template
    template = load_prompt_template("stress")
    prompt = template.replace("{{problem}}", problem)
    
    # Initialize OpenAI client with higher temperature for diversity
    client = OpenAIClient(model=model, concurrency=8)
    
    print(f"Generating {num_stress} stress test candidates (brute-force)...")
    
    # Generate brute-force solutions in parallel
    responses = await client.generate_multiple(prompt, num_stress, temperature=0.9)
    
    # Save each stress candidate to a file
    saved_count = 0
    for idx, response in enumerate(responses):
        if response:
            code = extract_code(response)
            output_path = Path(output_dir) / f"stress_{idx:03d}.py"
            with open(output_path, "w") as f:
                f.write(code)
            saved_count += 1
            print(f"  Saved: {output_path}")
    
    print(f"Generated {saved_count}/{num_stress} stress candidates")
    return saved_count


async def main():
    """Example usage."""
    problem = """
Given an array of integers, find the maximum sum of any contiguous subarray.

Input: First line contains n (1 ≤ n ≤ 10^5), the size of the array.
Second line contains n space-separated integers (-10^9 ≤ a[i] ≤ 10^9).

Output: Print the maximum subarray sum.

Example:
Input:
5
-2 1 -3 4 -1

Output:
4
"""
    
    await generate_stress_candidates(problem, "./output/stress", num_stress=3)


if __name__ == "__main__":
    asyncio.run(main())
