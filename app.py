"""Simple dashboard to inspect SECOM PoC outputs."""
from __future__ import annotations

from pathlib import Path

from secom.results import load_results

try:  # pragma: no cover - only used when rendering dashboard
    import streamlit as st
except Exception:
    st = None


def main() -> None:
    if st is None:
        raise RuntimeError("streamlit is not installed; run `uv pip install streamlit`")

    st.set_page_config(page_title="SECOM PoC", layout="wide")
    st.title("SECOM Semiconductor PoC Dashboard")

    rows = load_results()
    if not rows:
        st.info("No results found. Run `python experiments/run_all.py` first.")
        return

    st.subheader("Experiment table")
    st.dataframe(rows)

    st.subheader("Top results by PR-AUC")
    top = sorted(rows, key=lambda r: r.get("pooled_pr_auc", 0.0), reverse=True)[:10]
    st.table(top)

    st.subheader("Artifacts")
    base = Path("artifacts")
    if base.exists():
        st.write("Artifacts directory:", str(base.resolve()))
    for p in sorted(base.glob("*.png")) if base.exists() else []:
        st.image(str(p))


if __name__ == "__main__":
    main()
