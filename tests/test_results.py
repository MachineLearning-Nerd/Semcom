from __future__ import annotations

from secom.results import log_result, render_table


def test_log_and_render(tmp_path):
    p = tmp_path / "r.jsonl"
    log_result({"hypothesis": "H1", "name": "wf", "pooled_pr_auc": 0.18}, path=p)
    log_result({"hypothesis": "H1", "name": "random", "pooled_pr_auc": 0.31}, path=p)
    md = render_table(path=p)
    assert "H1" in md and "0.18" in md and "0.31" in md
