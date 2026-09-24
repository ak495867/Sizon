"""Dependency-free local Studio scaffold; optional UI dependencies can wrap these APIs."""
from __future__ import annotations
from pathlib import Path
from sizon.platform.report import build_report
def launch(run_dir="runs",port=8501):
    run_dir=Path(run_dir); report=build_report(run_dir); return {"status":"ready","report":str(report),"port":port,"next":"serve report.html with a local web server or connect a Streamlit frontend"}
