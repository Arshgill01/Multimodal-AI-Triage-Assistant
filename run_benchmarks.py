#!/usr/bin/env python3
"""
🧊 Frostbyte — Latency Benchmark Suite

Compares inference latency between:
  1. Rust (Axum + LightGBM FFI)  — port 3001
  2. Python (Flask + LightGBM)   — port 5001

Requirements:
  - Both servers running before executing this script
  - pip install requests tabulate matplotlib

Usage:
  python run_benchmarks.py
"""

import json
import statistics
import sys
import time
from datetime import datetime

import requests

# ── Config ────────────────────────────────────────────────────

RUST_URL = "http://localhost:3001/predict"
PYTHON_URL = "http://localhost:5001/predict"
WARMUP_ROUNDS = 5
BENCHMARK_ROUNDS = 50

# ── Test Patients ─────────────────────────────────────────────

TEST_PATIENTS = [
    {
        "name": "Critical — Unresponsive",
        "data": {
            "age": 72, "heart_rate": 145, "resp_rate": 38, "spo2": 78,
            "temp_f": 101.2, "systolic_bp": 72, "pain_scale": 0,
            "chief_complaint": "Unresponsive, found on floor",
        },
    },
    {
        "name": "Moderate — Chest Pain",
        "data": {
            "age": 55, "heart_rate": 110, "resp_rate": 22, "spo2": 94,
            "temp_f": 98.6, "systolic_bp": 140, "pain_scale": 8,
            "chief_complaint": "Severe chest pain radiating to left arm",
        },
    },
    {
        "name": "Minor — Ankle Sprain",
        "data": {
            "age": 28, "heart_rate": 78, "resp_rate": 14, "spo2": 99,
            "temp_f": 98.2, "systolic_bp": 118, "pain_scale": 4,
            "chief_complaint": "Twisted ankle while running",
        },
    },
]


def benchmark_endpoint(url: str, patient_data: dict, rounds: int) -> list[float]:
    """Hit the endpoint `rounds` times and return latencies in ms."""
    latencies = []
    for _ in range(rounds):
        start = time.perf_counter()
        resp = requests.post(url, json=patient_data, timeout=30)
        end = time.perf_counter()
        resp.raise_for_status()
        latencies.append((end - start) * 1000)  # ms
    return latencies


def check_server(url: str, name: str) -> bool:
    """Check if a server is reachable."""
    try:
        requests.get(url.replace("/predict", "/health"), timeout=3)
        return True
    except requests.ConnectionError:
        print(f"❌ {name} server not reachable at {url}")
        return False


def main():
    print("=" * 65)
    print("  🧊 FROSTBYTE LATENCY BENCHMARK SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Warmup: {WARMUP_ROUNDS} rounds | Benchmark: {BENCHMARK_ROUNDS} rounds")
    print("=" * 65)

    # ── Check servers ──
    rust_available = check_server(RUST_URL, "Rust")
    python_available = check_server(PYTHON_URL, "Python")

    if not rust_available and not python_available:
        print("\n❌ No servers available. Start at least one server.")
        sys.exit(1)

    results = {}

    for patient in TEST_PATIENTS:
        print(f"\n{'─' * 65}")
        print(f"  Patient: {patient['name']}")
        print(f"{'─' * 65}")

        results[patient["name"]] = {}

        for label, url, available in [
            ("Rust (Axum + FFI)", RUST_URL, rust_available),
            ("Python (Flask)", PYTHON_URL, python_available),
        ]:
            if not available:
                print(f"  {label}: SKIPPED (not running)")
                continue

            # Warmup
            print(f"  {label}: warming up...", end="", flush=True)
            try:
                benchmark_endpoint(url, patient["data"], WARMUP_ROUNDS)
                print(" done", flush=True)
            except Exception as e:
                print(f" FAILED: {e}")
                continue

            # Benchmark
            print(f"  {label}: benchmarking {BENCHMARK_ROUNDS} requests...", end="", flush=True)
            try:
                latencies = benchmark_endpoint(url, patient["data"], BENCHMARK_ROUNDS)
            except Exception as e:
                print(f" FAILED: {e}")
                continue
            print(" done", flush=True)

            avg = statistics.mean(latencies)
            med = statistics.median(latencies)
            p95 = sorted(latencies)[int(0.95 * len(latencies))]
            p99 = sorted(latencies)[int(0.99 * len(latencies))]
            mn = min(latencies)
            mx = max(latencies)

            results[patient["name"]][label] = {
                "avg": avg, "median": med, "p95": p95, "p99": p99,
                "min": mn, "max": mx, "samples": len(latencies),
            }

            print(f"    avg={avg:.2f}ms  median={med:.2f}ms  "
                  f"p95={p95:.2f}ms  p99={p99:.2f}ms  "
                  f"min={mn:.2f}ms  max={mx:.2f}ms")

    # ── Summary Table ──
    print(f"\n{'=' * 65}")
    print("  SUMMARY: End-to-End Latency Comparison")
    print(f"{'=' * 65}")
    print(f"\n{'Patient':<30} {'Rust (ms)':<15} {'Python (ms)':<15} {'Speedup':<10}")
    print("─" * 70)

    for patient_name, backends in results.items():
        rust_avg = backends.get("Rust (Axum + FFI)", {}).get("avg", None)
        python_avg = backends.get("Python (Flask)", {}).get("avg", None)

        rust_str = f"{rust_avg:.2f}" if rust_avg else "N/A"
        python_str = f"{python_avg:.2f}" if python_avg else "N/A"

        if rust_avg and python_avg:
            speedup = python_avg / rust_avg
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup_str = "—"

        short_name = patient_name[:28]
        print(f"{short_name:<30} {rust_str:<15} {python_str:<15} {speedup_str:<10}")

    # ── Save results ──
    report_path = "benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {report_path}")

    # ── Generate Markdown Report ──
    generate_markdown_report(results)


def generate_markdown_report(results: dict):
    """Generate BENCHMARKS.md for the hackathon pitch."""
    lines = [
        "# 🧊 Frostbyte — Inference Latency Benchmarks\n",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Rounds: {BENCHMARK_ROUNDS} per patient\n",
        "",
        "## Why Rust?\n",
        "Emergency departments process thousands of patients daily. Every millisecond",
        "of inference latency compounds. Our Rust backend performs native LightGBM",
        "inference via FFI, eliminating Python's GIL overhead and interpreter costs.\n",
        "",
        "## Results\n",
        "| Patient Scenario | Rust (Axum + FFI) | Python (Flask) | Speedup |",
        "|:---|:---:|:---:|:---:|",
    ]

    for patient_name, backends in results.items():
        rust = backends.get("Rust (Axum + FFI)", {})
        python = backends.get("Python (Flask)", {})

        rust_str = f"{rust.get('avg', 0):.2f}ms" if rust else "N/A"
        python_str = f"{python.get('avg', 0):.2f}ms" if python else "N/A"

        if rust and python:
            speedup = python["avg"] / rust["avg"]
            speedup_str = f"**{speedup:.1f}x**"
        else:
            speedup_str = "—"

        lines.append(f"| {patient_name} | {rust_str} | {python_str} | {speedup_str} |")

    lines.extend([
        "",
        "## Methodology\n",
        "- **Rust backend**: Axum HTTP server → LightGBM FFI inference (port 3001)",
        "- **Python baseline**: Flask → sklearn LightGBM predict (port 5001)",
        "- Both pipelines include ClinicalBERT embedding extraction + PCA",
        "- Measured end-to-end (HTTP request → JSON response), not just model inference",
        f"- {WARMUP_ROUNDS} warmup rounds discarded, {BENCHMARK_ROUNDS} measured rounds per patient\n",
        "",
        "## Detailed Metrics\n",
    ])

    for patient_name, backends in results.items():
        lines.append(f"### {patient_name}\n")
        lines.append("| Metric | Rust | Python |")
        lines.append("|:---|:---:|:---:|")

        for metric in ["avg", "median", "p95", "p99", "min", "max"]:
            rust_val = backends.get("Rust (Axum + FFI)", {}).get(metric)
            python_val = backends.get("Python (Flask)", {}).get(metric)
            r_str = f"{rust_val:.2f}ms" if rust_val else "—"
            p_str = f"{python_val:.2f}ms" if python_val else "—"
            lines.append(f"| {metric} | {r_str} | {p_str} |")
        lines.append("")

    report = "\n".join(lines)
    with open("BENCHMARKS.md", "w") as f:
        f.write(report)
    print("📊 BENCHMARKS.md generated for hackathon pitch")


if __name__ == "__main__":
    main()
