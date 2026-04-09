"""
Report — Structured experiment reporting (console + JSON).
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class TestCase:
    """A single test case within an experiment."""

    name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ExperimentReport:
    """Complete report for one experiment."""

    experiment_id: str
    experiment_name: str
    formal_reference: str
    claim: str
    status: str = "NOT_RUN"  # PASS, FAIL, ERROR, NOT_RUN
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    test_cases: List[TestCase] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_case(self, case: TestCase):
        """Add a test case to the report."""
        self.test_cases.append(case)
        self.total_cases += 1
        if case.passed:
            self.passed_cases += 1
        else:
            self.failed_cases += 1

    def finalize(self):
        """Set final status based on results."""
        if self.total_cases == 0:
            self.status = "NOT_RUN"
        elif self.failed_cases == 0:
            self.status = "PASS"
        else:
            self.status = "FAIL"


class ExperimentTimer:
    """Context manager for timing experiments."""

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time


def print_report(report: ExperimentReport):
    """Print a formatted report to console."""
    status_icon = "[PASS]" if report.status == "PASS" else "[FAIL]"

    print()
    print("=" * 72)
    print(f"  {status_icon}  {report.experiment_id}: {report.experiment_name}")
    print("=" * 72)
    print(f"  Formal Reference : {report.formal_reference}")
    print(f"  Claim            : {report.claim}")
    print(f"  Status           : {report.status}")
    print(f"  Cases            : {report.passed_cases}/{report.total_cases} passed")
    print(f"  Duration         : {report.duration_seconds:.3f}s")
    print("-" * 72)

    if report.failed_cases > 0:
        print("  FAILURES:")
        for case in report.test_cases:
            if not case.passed:
                print(f"    [x] {case.name}")
                if case.error:
                    print(f"      Error: {case.error}")
                for key, val in case.details.items():
                    val_str = str(val)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    print(f"      {key}: {val_str}")
        print("-" * 72)

    print()


def save_report_json(report: ExperimentReport, path: str):
    """Save report as JSON file."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict, handling numpy types
    def _serialize(obj):
        if hasattr(obj, "__array__"):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return {k: _serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        return obj

    data = _serialize(report)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def print_summary(reports: List[ExperimentReport]):
    """Print a master summary of all experiment reports."""
    print()
    print("=" * 72)
    print("  SYNAPSE EXPERIMENT SUITE — MASTER SUMMARY")
    print("=" * 72)

    total_pass = sum(1 for r in reports if r.status == "PASS")
    total_fail = sum(1 for r in reports if r.status == "FAIL")
    total_error = sum(1 for r in reports if r.status == "ERROR")
    total_skip = sum(1 for r in reports if r.status == "NOT_RUN")

    for r in reports:
        icon = "[PASS]" if r.status == "PASS" else ("[FAIL]" if r.status == "FAIL" else "[WARN]")
        cases = f"{r.passed_cases}/{r.total_cases}"
        print(f"  {icon}  {r.experiment_id:<8} {r.experiment_name:<45} {cases:>8}  {r.duration_seconds:>7.2f}s")

    print("-" * 72)
    print(f"  Total: {len(reports)} experiments  |  "
          f"PASS {total_pass}  |  FAIL {total_fail}  |  "
          f"ERROR {total_error}  |  SKIP {total_skip}")

    all_passed = total_fail == 0 and total_error == 0 and total_skip == 0
    if all_passed:
        print()
        print("  >>> ALL EXPERIMENTS PASSED -- FORMAL CLAIMS VERIFIED")
        print("  --> Application implementations may proceed.")
    else:
        print()
        print("  [!] GATE CHECK FAILED -- Do NOT proceed to application implementation.")
        print("  --> Fix synapse_core/ until all experiments pass.")

    print("=" * 72)
    print()

    return all_passed
