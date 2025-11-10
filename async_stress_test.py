import asyncio
import tempfile
import uuid
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os
from collections import Counter, defaultdict


@dataclass
class Problem:
    title: str
    description: str
    sample_tests: List[Tuple[str, str]]  # (input, expected_output)
    additional_tests: List[str]  # only inputs


@dataclass
class Candidate:
    id: str
    code: str


class LLMClient:
    """Simple async LLM client interface. By default uses a mock generator. Replace
    `generate_candidate` with real API calls (OpenAI, Anthropic, etc.) as needed.
    """

    def __init__(self, concurrency: int = 6):
        self._sem = asyncio.Semaphore(concurrency)

    async def generate_candidate(self, problem: Problem, role: str = "solution", context: Optional[str] = None) -> Candidate:
        # Acquire semaphore to limit concurrent API calls
        async with self._sem:
            # Mock generation: produce trivial Python program echoing input or
            # naive brute-force stress candidate depending on `role`.
            await asyncio.sleep(0.05)  # simulate latency
            if role == "stress":
                code = (
                    "import sys\n"
                    "data = sys.stdin.read()\n"
                    "# Mock brute-force / stress program: print fallback\n"
                    "print(data.strip())\n"
                )
            else:
                code = (
                    "import sys\n"
                    "inp = sys.stdin.read()\n"
                    "# Mock candidate: just echo input (replace with LLM output)\n"
                    "print(inp.strip())\n"
                )
            return Candidate(id=str(uuid.uuid4()), code=code)


class CodeExecutor:
    """Execute Python code asynchronously with per-test timeouts. Writes code to
    a temporary file and runs it with `python3`.
    WARNING: Executing untrusted code is dangerous. Run inside a sandbox/container.
    """

    def __init__(self, max_concurrency: int = 12):
        self._sem = asyncio.Semaphore(max_concurrency)

    async def run(self, candidate: Candidate, input_data: str, timeout: float) -> Tuple[bool, str]:
        async with self._sem:
            # write to temp file
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(candidate.code)
                path = f.name

            proc = await asyncio.create_subprocess_exec(
                "python3",
                path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(input=input_data.encode()), timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                os.remove(path)
                return False, "__TIMEOUT__"

            os.remove(path)

            if proc.returncode != 0:
                return False, (stderr.decode().strip() or "__ERROR__")

            return True, stdout.decode().strip()


class StressTester:
    def __init__(self, llm: LLMClient, executor: CodeExecutor, max_debug_attempts: int = 3):
        self.llm = llm
        self.executor = executor
        self.max_debug_attempts = max_debug_attempts

    async def generate_candidates(self, problem: Problem, n: int, role: str = "solution") -> List[Candidate]:
        tasks = [self.llm.generate_candidate(problem, role=role) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def self_debug(self, candidate: Candidate, problem: Problem, timeout: float) -> Optional[Candidate]:
        # Run candidate against sample tests; if any fail, ask for revision up to attempts
        attempt = 0
        current_candidate = candidate
        while attempt <= self.max_debug_attempts:
            # run all sample tests in parallel
            runs = [self.executor.run(current_candidate, inp, timeout) for inp, _ in problem.sample_tests]
            results = await asyncio.gather(*runs)

            all_ok = True
            for (ok, out), (_, expected) in zip(results, problem.sample_tests):
                if not ok or out.strip() != expected.strip():
                    all_ok = False
                    break

            if all_ok:
                return current_candidate

            # ask LLM to debug (here we regenerate; replace with targeted debugging prompt)
            attempt += 1
            current_candidate = await self.llm.generate_candidate(problem, role="solution", context=f"debug attempt {attempt}")

        return None

    async def build_stress_predicted_subset(self, stress_candidates: List[Candidate], additional_inputs: List[str], timeout: float, min_agree: int = 2) -> Dict[int, str]:
        # For each stress candidate, run all additional inputs and collect outputs
        # returns mapping idx->predicted_output for reliable tests
        # Run stress candidate executions concurrently per input
        # results_by_test[idx] -> list of outputs
        results_by_test: Dict[int, List[str]] = defaultdict(list)

        async def run_one(stress_candidate: Candidate):
            runs = [self.executor.run(stress_candidate, inp, timeout) for inp in additional_inputs]
            res = await asyncio.gather(*runs)
            for i, (ok, out) in enumerate(res):
                if ok:
                    results_by_test[i].append(out)

        await asyncio.gather(*(run_one(c) for c in stress_candidates))

        # pick tests with mode count >= min_agree
        stress_predicted: Dict[int, str] = {}
        for idx, outs in results_by_test.items():
            if not outs:
                continue
            c = Counter(outs)
            val, count = c.most_common(1)[0]
            if count >= min_agree:
                stress_predicted[idx] = val

        return stress_predicted

    async def filter_candidates_by_tests(self, candidates: List[Candidate], tests: Dict[int, str], additional_inputs: List[str], timeout: float) -> List[Candidate]:
        # tests: idx->expected_output
        if not tests:
            return candidates

        async def candidate_ok(cand: Candidate) -> Optional[Candidate]:
            runs = [self.executor.run(cand, additional_inputs[i], timeout) for i in tests.keys()]
            res = await asyncio.gather(*runs)
            for (ok, out), idx in zip(res, tests.keys()):
                if (not ok) or out != tests[idx]:
                    return None
            return cand

        filtered = await asyncio.gather(*(candidate_ok(c) for c in candidates))
        return [c for c in filtered if c is not None]

    async def majority_vote(self, candidates: List[Candidate], additional_inputs: List[str], timeout: float) -> Optional[Candidate]:
        # cluster by outputs over all additional tests, pick candidate from largest cluster
        if not candidates:
            return None

        async def run_all_outputs(cand: Candidate):
            runs = [self.executor.run(cand, inp, timeout) for inp in additional_inputs]
            res = await asyncio.gather(*runs)
            # return tuple of outputs (or __ERR__ markers)
            outputs = tuple((out if ok else "__ERR__") for ok, out in res)
            return cand, outputs

        rows = await asyncio.gather(*(run_all_outputs(c) for c in candidates))
        cluster: Dict[Tuple[str, ...], List[Candidate]] = defaultdict(list)
        for cand, outs in rows:
            cluster[outs].append(cand)

        # choose largest cluster
        best_cluster = max(cluster.items(), key=lambda kv: len(kv[1]))
        chosen = best_cluster[1][0]
        return chosen


async def main():
    # Example usage with a mock problem. Replace LLMClient.generate_candidate with
    # a real async OpenAI/Anthropic client to use real models.
    problem = Problem(
        title="Mock Echo",
        description="Echo input",
        sample_tests=[("hello\n", "hello")],
        additional_tests=["abc\n", "123\n", "hello\n"],
    )

    llm = LLMClient(concurrency=4)
    executor = CodeExecutor(max_concurrency=8)
    tester = StressTester(llm, executor, max_debug_attempts=2)

    # 1) Generate candidate solutions in parallel
    candidates = await tester.generate_candidates(problem, n=6, role="solution")

    # 2) Self-debug each candidate (in parallel)
    debugged = await asyncio.gather(*(tester.self_debug(c, problem, timeout=2.0) for c in candidates))
    debugged = [c for c in debugged if c is not None]

    # 3) Generate stress (brute-force) candidates
    stress_candidates = await tester.generate_candidates(problem, n=4, role="stress")

    # 4) Build stress-predicted subset
    stress_predicted = await tester.build_stress_predicted_subset(stress_candidates, problem.additional_tests, timeout=1.0, min_agree=2)

    # 5) Filter debugged candidates by stress-predicted tests
    filtered = await tester.filter_candidates_by_tests(debugged, stress_predicted, problem.additional_tests, timeout=1.0)

    # 6) Majority vote among remaining candidates
    final = await tester.majority_vote(filtered or debugged, problem.additional_tests, timeout=1.0)

    print("--- Summary ---")
    print("Candidates generated:", len(candidates))
    print("After self-debug:", len(debugged))
    print("Stress predicted tests:", stress_predicted)
    print("After stress filtering:", len(filtered))
    print("Final candidate id:", final.id if final else None)


if __name__ == "__main__":
    asyncio.run(main())
