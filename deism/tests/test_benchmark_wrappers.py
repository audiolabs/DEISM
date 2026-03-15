from pathlib import Path


def test_benchmark_wrapper_scripts_exist():
    root = Path(__file__).resolve().parents[2]
    assert (root / "examples/benchmarks/run_speed_suite.py").exists()
    assert (root / "examples/benchmarks/run_accuracy_suite.py").exists()
    assert (root / "tools/run_acceleration_gates.py").exists()
