"""
Generate additional brute-force test inputs using the LLM.
Saves generated test inputs into output directory for use in stress testing.
"""
import asyncio
import os
from pathlib import Path
from typing import List
from llm_client import OpenAIClient


def load_prompt_template(template_name: str) -> str:
    prompt_path = Path(__file__).parent / "prompts" / f"{template_name}.prompt"
    with open(prompt_path, "r") as f:
        return f.read()


def sanitize_response(resp: str) -> List[str]:
    # Split by lines and strip; ignore empty lines
    lines = [line.strip() for line in resp.splitlines()]
    return [line for line in lines if line]


async def generate_tests(problem: str, output_dir: str, num: int = 5, model: str = "gpt-4o") -> List[str]:
    """
    Generate `num` test-case inputs for the given problem and save to output_dir/generated_tests.json
    Returns list of generated input strings.
    """
    os.makedirs(output_dir, exist_ok=True)
    template = load_prompt_template("generator")
    prompt = template.replace("{{problem}}", problem).replace("{{num}}", str(num))

    client = OpenAIClient(model=model, concurrency=6)
    print(f"Generating {num} additional test inputs via LLM...")
    responses = await client.generate_multiple(prompt, num, temperature=0.9)

    tests: List[str] = []
    for idx, resp in enumerate(responses):
        if not resp:
            continue
        lines = sanitize_response(resp)
        # Join multiple lines with literal \n so each test case is a single-line string representing stdin
        test_input = "\\n".join(lines)
        if not test_input.endswith("\\n"):
            test_input = test_input + "\\n"
        tests.append(test_input)
        # save each test to a file for inspection
        with open(Path(output_dir) / f"gen_test_{idx:03d}.txt", "w") as f:
            f.write(test_input)
        print(f"  Saved generated test: gen_test_{idx:03d}.txt")

    # Save an index file listing all
    import json
    with open(Path(output_dir) / "generated_tests.json", "w") as f:
        json.dump([{"input": t} for t in tests], f, indent=2)

    print(f"Generated {len(tests)}/{num} tests")
    return tests


if __name__ == "__main__":
    # quick manual test
    problem = """
Given an array of integers, find the maximum sum of any contiguous subarray.

Input: First line contains n (1 ≤ n ≤ 10^5), the size of the array.
Second line contains n space-separated integers.
"""
    asyncio.run(generate_tests(problem, "./output/generated", num=4))
