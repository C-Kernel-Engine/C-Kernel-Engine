"""
Base classes for the staged kernel validation system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json
import sys


# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestResult:
    """Result of a single test within a stage"""
    name: str
    status: TestStatus
    message: str = ""
    max_diff: Optional[float] = None
    mean_diff: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status in (TestStatus.PASS, TestStatus.SKIP)

    def __str__(self) -> str:
        status_colors = {
            TestStatus.PASS: GREEN,
            TestStatus.FAIL: RED,
            TestStatus.SKIP: YELLOW,
            TestStatus.ERROR: RED,
        }
        color = status_colors.get(self.status, RESET)
        diff_str = ""
        if self.max_diff is not None:
            diff_str = f" (max_diff={self.max_diff:.2e})"
        return f"{self.name}: {color}{self.status.value}{RESET}{diff_str}"


@dataclass
class StageResult:
    """Result of a validation stage"""
    stage_num: int
    stage_name: str
    tests: List[TestResult] = field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def passed(self) -> bool:
        if self.error_message:
            return False
        return all(t.passed for t in self.tests)

    @property
    def num_passed(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.PASS)

    @property
    def num_failed(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.FAIL)

    @property
    def num_skipped(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.SKIP)

    def add_test(self, result: TestResult):
        self.tests.append(result)

    def first_failure(self) -> Optional[TestResult]:
        """Get the first failing test"""
        for t in self.tests:
            if t.status == TestStatus.FAIL:
                return t
        return None

    def print_summary(self, verbose: bool = False):
        """Print stage summary"""
        status = f"{GREEN}PASSED{RESET}" if self.passed else f"{RED}FAILED{RESET}"
        print(f"\n{BOLD}Stage {self.stage_num}: {self.stage_name}{RESET} - {status}")
        print("=" * 60)

        if self.error_message:
            print(f"  {RED}Error: {self.error_message}{RESET}")
            return

        for test in self.tests:
            print(f"  {test}")
            if verbose and test.details:
                for k, v in test.details.items():
                    print(f"    {k}: {v}")

        print(f"\n  Summary: {self.num_passed} passed, {self.num_failed} failed, {self.num_skipped} skipped")

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'stage_num': self.stage_num,
            'stage_name': self.stage_name,
            'passed': self.passed,
            'tests': [
                {
                    'name': t.name,
                    'status': t.status.value,
                    'message': t.message,
                    'max_diff': t.max_diff,
                    'details': t.details,
                }
                for t in self.tests
            ],
            'error_message': self.error_message,
        }


@dataclass
class ValidationReport:
    """Complete validation report across all stages"""
    stages: Dict[int, StageResult] = field(default_factory=dict)
    gated_at: Optional[int] = None  # Stage where validation stopped due to failure

    def add_stage_result(self, stage_num: int, result: StageResult):
        self.stages[stage_num] = result

    def all_passed(self) -> bool:
        return all(s.passed for s in self.stages.values())

    def first_failure_stage(self) -> Optional[int]:
        """Get the first stage that failed"""
        for num in sorted(self.stages.keys()):
            if not self.stages[num].passed:
                return num
        return None

    def print_summary(self):
        """Print complete validation report"""
        print("\n" + "=" * 60)
        print(f"{BOLD}VALIDATION SUMMARY{RESET}")
        print("=" * 60)

        for num in sorted(self.stages.keys()):
            stage = self.stages[num]
            status = f"{GREEN}PASSED{RESET}" if stage.passed else f"{RED}FAILED{RESET}"
            print(f"  Stage {num} ({stage.stage_name}): {status}")

            if not stage.passed:
                failure = stage.first_failure()
                if failure:
                    print(f"    First failure: {failure.name}")
                    if failure.message:
                        print(f"    Message: {failure.message}")

        if self.gated_at:
            print(f"\n{YELLOW}Validation gated at Stage {self.gated_at}{RESET}")

        overall = f"{GREEN}ALL STAGES PASSED{RESET}" if self.all_passed() else f"{RED}VALIDATION FAILED{RESET}"
        print(f"\n{BOLD}Overall: {overall}{RESET}")

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string"""
        data = {
            'all_passed': self.all_passed(),
            'gated_at': self.gated_at,
            'stages': {
                str(num): stage.to_dict()
                for num, stage in self.stages.items()
            }
        }
        return json.dumps(data, indent=indent)

    def save_json(self, path: str):
        """Save report to JSON file"""
        with open(path, 'w') as f:
            f.write(self.to_json())


class BaseValidator:
    """Base class for stage validators"""

    def __init__(self, gguf_path: str, bump_path: Optional[str] = None,
                 manifest_path: Optional[str] = None, verbose: bool = False):
        self.gguf_path = gguf_path
        self.bump_path = bump_path
        self.manifest_path = manifest_path
        self.verbose = verbose

    def log(self, message: str):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(f"  [DEBUG] {message}")

    def run(self) -> StageResult:
        """Run validation - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement run()")
