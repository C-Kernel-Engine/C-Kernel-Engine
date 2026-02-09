#!/usr/bin/env python3
"""
Compare two performance profiles.
Detects regressions and improvements between baseline and current profiles.
"""
import json
import sys


def compare(baseline: dict, current: dict, threshold: float = 5.0) -> dict:
    """Compare profiles and detect regressions."""
    results = {
        "regressions": [],
        "improvements": [],
        "summary": ""
    }

    baseline_map = {k["kernel"]: k for k in baseline["kernels"]}
    current_map = {k["kernel"]: k for k in current["kernels"]}

    all_names = set(baseline_map.keys()) | set(current_map.keys())

    total_regression = 0.0
    total_improvement = 0.0

    for name in sorted(all_names):
        b = baseline_map.get(name)
        c = current_map.get(name)

        if b and c:
            diff = c["time_ms"] - b["time_ms"]
            pct = (diff / b["time_ms"]) * 100 if b["time_ms"] > 0 else 0

            if pct > threshold:
                results["regressions"].append({
                    "kernel": name,
                    "baseline_ms": b["time_ms"],
                    "current_ms": c["time_ms"],
                    "regression_pct": round(pct, 2)
                })
                total_regression += diff
            elif pct < -threshold:
                results["improvements"].append({
                    "kernel": name,
                    "baseline_ms": b["time_ms"],
                    "current_ms": c["time_ms"],
                    "improvement_pct": round(-pct, 2)
                })
                total_improvement += -diff

    baseline_total = baseline["total_time_ms"]
    current_total = current["total_time_ms"]
    overall_change = ((current_total - baseline_total) / baseline_total) * 100 if baseline_total > 0 else 0

    results["summary"] = {
        "baseline_total_ms": round(baseline_total, 2),
        "current_total_ms": round(current_total, 2),
        "overall_change_pct": round(overall_change, 2),
        "regression_count": len(results["regressions"]),
        "improvement_count": len(results["improvements"]),
    }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare performance profiles"
    )
    parser.add_argument("baseline", help="Baseline JSON profile file")
    parser.add_argument("current", help="Current JSON profile file")
    parser.add_argument("--threshold", type=float, default=5.0,
                       help="Regression threshold %% (default: 5.0)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.current) as f:
        current = json.load(f)

    results = compare(baseline, current, args.threshold)

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        return 0

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    summary = results["summary"]
    print(f"\nBaseline: {summary['baseline_total_ms']:.2f} ms")
    print(f"Current:  {summary['current_total_ms']:.2f} ms")
    change = summary['overall_change_pct']
    sign = "+" if change >= 0 else ""
    print(f"Overall:  {sign}{change:.1f}%")

    print("\n" + "-" * 60)

    if results['regressions']:
        print(f"\nREGRESSIONS ({len(results['regressions'])}):")
        print("-" * 40)
        for r in sorted(results['regressions'], key=lambda x: -x['regression_pct']):
            print(f"  {r['kernel']}:")
            print(f"    {r['baseline_ms']:.2f} -> {r['current_ms']:.2f} ms (+{r['regression_pct']:.1f}%)")

    if results['improvements']:
        print(f"\nIMPROVEMENTS ({len(results['improvements'])}):")
        print("-" * 40)
        for i in sorted(results['improvements'], key=lambda x: -x['improvement_pct']):
            print(f"  {i['kernel']}:")
            print(f"    {i['baseline_ms']:.2f} -> {i['current_ms']:.2f} ms (-{i['improvement_pct']:.1f}%)")

    print("\n" + "=" * 60)

    if results['regressions']:
        print(f"\nWARNING: {len(results['regressions'])} regressions exceed {args.threshold}% threshold")
        sys.exit(1)
    else:
        print(f"\nAll kernels within {args.threshold}% threshold")
        print("No regressions detected - performance is stable or improved.")
        sys.exit(0)


if __name__ == "__main__":
    main()
