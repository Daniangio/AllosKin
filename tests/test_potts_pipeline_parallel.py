import time
import os

os.environ["PHASE_DATA_ROOT"] = "/tmp/phase-test-data"

from phase.potts import pipeline


def _ordered_worker(payload):
    time.sleep(float(payload["delay"]))
    return {"row": int(payload["row"])}


def test_run_chain_batch_preserves_input_order_with_parallel_workers():
    payloads = [
        {"row": 0, "delay": 0.20},
        {"row": 1, "delay": 0.01},
        {"row": 2, "delay": 0.10},
    ]

    out = pipeline._run_chain_batch(
        payloads,
        worker_fn=_ordered_worker,
        max_workers=2,
        report=None,
        report_message="Chains {current}/{total}",
        report_start=0,
        report_span=10,
    )

    assert [row["row"] for row in out] == [0, 1, 2]


def test_run_chain_batch_reports_progress_and_quiet_prints(monkeypatch, capsys):
    calls = {"max_workers": None}
    reports = []

    def fake_run_local(payloads, worker_fn, max_workers, progress_callback=None, progress_label=None):
        calls["max_workers"] = max_workers
        if progress_callback is not None:
            progress_callback(progress_label or "batch", 1, len(payloads))
            progress_callback(progress_label or "batch", len(payloads), len(payloads))
        return [{"row": idx} for idx, _ in enumerate(payloads)]

    monkeypatch.setattr(pipeline, "run_local_payload_batch", fake_run_local)

    out = pipeline._run_chain_batch(
        [{"row": 0}, {"row": 1}],
        worker_fn=lambda payload: payload,
        max_workers=4,
        report=lambda message, pct: reports.append((message, pct)),
        report_message="Replica exchange chains {current}/{total}",
        report_start=50,
        report_span=10,
        quiet_print_prefix="[rex] chain",
        quiet_mode=True,
    )

    assert calls["max_workers"] == 4
    assert out == [{"row": 0}, {"row": 1}]
    assert reports == [
        ("Replica exchange chains 1/2", 55.0),
        ("Replica exchange chains 2/2", 60.0),
    ]
    stdout = capsys.readouterr().out
    assert "[rex] chain 1/2 complete" in stdout
    assert "[rex] chain 2/2 complete" in stdout
