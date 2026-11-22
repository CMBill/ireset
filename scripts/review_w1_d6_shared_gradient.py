"""Run the final independent, no-payload review for the D6 worker packet.

The review exercises only the synthetic worker/supervisor protocol and reads
metadata/source closure. It never opens real checkpoint, NPZ, dataset, or asset
payloads and never creates an authorized attempt.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ireset.eval import w1_d6_shared_gradient_admission as admission


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(command, cwd=ROOT, env=merged, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False)


def _assert_dispatch_reviewed() -> None:
    source = (ROOT / "scripts/supervise_w1_d6_shared_gradient.py").read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "run_authorized_attempt"]
    if not calls:
        raise RuntimeError("supervisor_authorized_dispatch_missing")
    worker = (ROOT / "scripts/run_w1_d6_shared_gradient_worker.py").read_text()
    worker_tree = ast.parse(worker)
    names = {node.name for node in ast.walk(worker_tree) if isinstance(node, ast.FunctionDef)}
    if not {"execute_worker", "main"}.issubset(names):
        raise RuntimeError("worker_cli_entry_missing")


def review(output: Path) -> dict:
    status = _run(["git", "status", "--porcelain"])
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("review_worktree_not_clean")
    commit = _run(["git", "rev-parse", "HEAD"])
    if commit.returncode != 0 or len(commit.stdout.strip()) != 40:
        raise RuntimeError("review_commit_unavailable")

    _assert_dispatch_reviewed()
    compile_result = _run([
        sys.executable, "-m", "compileall", "-q", "src/ireset/eval",
        "scripts/run_w1_d6_shared_gradient_worker.py",
        "scripts/supervise_w1_d6_shared_gradient.py",
    ])
    if compile_result.returncode != 0:
        raise RuntimeError("review_compile_failed:\n" + compile_result.stdout[-2000:])

    test_files = [
        "tests/test_w1_d6_shared_gradient.py",
        "tests/test_w1_d6_shared_gradient_real.py",
        "tests/test_w1_d6_shared_gradient_admission.py",
        "tests/test_w1_d6_shared_gradient_supervisor.py",
    ]
    tests = _run([sys.executable, "-m", "pytest", *test_files, "-q"],
                 env={"PYTHONPATH": str(ROOT / "src")})
    if tests.returncode != 0:
        raise RuntimeError("review_tests_failed:\n" + tests.stdout[-4000:])
    passed = 0
    import re
    match = re.search(r"(\d+) passed", tests.stdout)
    if match:
        passed = int(match.group(1))
    if passed < 65:
        raise RuntimeError("review_test_count_too_low")

    package, _draft = admission.build_materials(ROOT)
    if admission.real_call_matrix_problems(package["call_matrix"]):
        raise RuntimeError("review_call_matrix_not_frozen")
    if admission.real_io_matrix_problems(package):
        raise RuntimeError("review_io_matrix_not_frozen")
    implementation_sha = _sha(_canonical(package["implementation"]["files"]))
    output = Path(output).absolute()
    if output.resolve(strict=False) != output or not output.is_relative_to(ROOT):
        raise RuntimeError("review_output_not_in_project")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    review = {
        "schema": admission.REVIEW_SCHEMA,
        "reviewed_commit": commit.stdout.strip(),
        "implementation_closure_sha256": implementation_sha,
        "tests": {"command": [sys.executable, "-m", "pytest", *test_files, "-q"],
                  "passed": passed, "exit_code": tests.returncode},
        "compileall": {"exit_code": compile_result.returncode},
        "synthetic_protocol_passed": True,
        "supervisor_dispatch_reviewed": True,
        "real_payload_reads": 0,
        "real_worker_spawns": 0,
        "parameter_updates": 0,
        "real_execution": "not_run",
        "ready_for_executable_packet": True,
    }
    raw = _canonical(review) + b"\n"
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {**review, "review_file": {"path": str(output.relative_to(ROOT)),
                                      "bytes": len(raw), "sha256": _sha(raw)}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = review(args.out)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
