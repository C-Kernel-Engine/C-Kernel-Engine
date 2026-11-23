"""Reject added OpenMP code in first-party sources and build configuration."""
import argparse
import re
import subprocess

OPENMP = re.compile(
    r"\b_OPENMP\b|\bomp\.h\b|\b(?:pragma\s+|_Pragma\s*\(\s*\"|__pragma\s*\(\s*)omp\b"
    r"|\bomp_(?:[a-zA-Z0-9_]+\s*\(|[A-Z0-9_]+\b)|\b(?:GOMP|__kmpc)_\w+"
    r"|-fopenmp\b|-qopenmp\b|/openmp\b|-l(?:gomp|omp|iomp5)\b"
    r"|\bOpenMP::|find_package\(\s*OpenMP|\bopenmp\s*=\s*(?!False\b|false\b)\S"
)

PATHS = [
    "src/", "include/", "version/v8/", "scripts/", "server/", "tests/", "unittest/",
    "Makefile", "*.mk", "CMakeLists.txt", "*.cmake", ".github/workflows/",
    ":(exclude)scripts/check_openmp_policy.py", ":(exclude)tests/test_openmp_policy.py",
    ":(exclude)**/*.md", ":(exclude)**/*.html",
    ":(exclude)**/fixtures/**",
]


def violations(diff):
    path = ""
    line = 0
    for row in diff.splitlines():
        if row.startswith("+++ b/"):
            path = row[6:]
        elif row.startswith("@@ "):
            line = int(re.search(r" \+(\d+)", row)[1])
        elif row.startswith("+"):
            if OPENMP.search(row[1:]):
                yield f"{path}:{line}: {row[1:].strip()}"
            line += 1
        elif row.startswith(" "):
            line += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("head", nargs="?", default="HEAD")
    args = parser.parse_args()
    diff = subprocess.check_output(
        ["git", "diff", "--no-color", "--no-ext-diff", "--unified=0",
         f"{args.base}...{args.head}", "--", *PATHS], text=True, errors="replace"
    )
    findings = list(violations(diff))
    if findings:
        print("Use CKE thread-pool dispatch for new parallel execution.")
        print("\n".join(findings))
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
