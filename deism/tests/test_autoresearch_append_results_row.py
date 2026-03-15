from tools.autoresearch.append_results_row import summarize_report


def test_summarize_report_handles_image_matrix_rows():
    report = {
        "n_cases": 2,
        "n_failures": 0,
        "threshold_breached": False,
        "rows": [
            {
                "image_gen_speedup": 1.2,
                "candidate_image_backend": "rewrite_cpu",
                "strict_equivalence_passed": True,
            },
            {
                "image_gen_speedup": 0.8,
                "candidate_image_backend": "rewrite_cpu",
                "strict_equivalence_passed": True,
            },
        ],
    }

    summary = summarize_report(report)

    assert summary["eval_passed"] is True
    assert summary["accuracy_gate_passed"] is True
    assert summary["accel_accuracy_gate_passed"] is True
    assert summary["median_speedup"] == "1.0"
    assert summary["backend_set"] == "rewrite_cpu"
