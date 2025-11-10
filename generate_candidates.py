"""
Generate candidate solutions for a problem.
Creates multiple solution files in the specified output directory.
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


async def generate_candidates(problem: str, output_dir: str, num_candidates: int = 9, model: str = "gpt-4o"):
    """
    Generate multiple candidate solutions in parallel.
    
    Args:
        problem: Problem description
        output_dir: Directory to save candidate files
        num_candidates: Number of candidates to generate
        model: OpenAI model to use
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prompt template
    template = load_prompt_template("candidate")
    prompt = template.replace("{{problem}}", problem)
    
    # Initialize OpenAI client
    client = OpenAIClient(model=model, concurrency=10)
    
    print(f"Generating {num_candidates} candidate solutions...")
    
    # Generate solutions in parallel
    responses = await client.generate_multiple(prompt, num_candidates, temperature=0.8)
    
    # Save each candidate to a file
    saved_count = 0
    for idx, response in enumerate(responses):
        if response:
            code = extract_code(response)
            output_path = Path(output_dir) / f"candidate_{idx:03d}.py"
            with open(output_path, "w") as f:
                f.write(code)
            saved_count += 1
            print(f"  Saved: {output_path}")
    
    print(f"Generated {saved_count}/{num_candidates} candidates")
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
    
    await generate_candidates(problem, "./output/candidates", num_candidates=5)


if __name__ == "__main__":
    asyncio.run(main())
