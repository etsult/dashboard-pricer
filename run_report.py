#!/usr/bin/env python3
"""
CLI entry point for the pricer consistency benchmark report.

Usage:
  python run_report.py
  python run_report.py --n-formula 1000 --n-book 400 --n-nn 1000 --output report.html

Opens the report in the default browser after generation.
"""

import argparse
import webbrowser
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from tests.ir.report import run_full_report

def main():
    parser = argparse.ArgumentParser(
        description="Generate pricer consistency benchmark report"
    )
    parser.add_argument("--n-formula", type=int, default=500,
                        help="Formula-level samples per scenario (default 500)")
    parser.add_argument("--n-book",    type=int, default=200,
                        help="Book positions per product/CCY group (default 200)")
    parser.add_argument("--n-nn",      type=int, default=500,
                        help="NN samples per convention (default 500)")
    parser.add_argument("--output",    type=str, default="pricer_report.html",
                        help="Output HTML file path (default pricer_report.html)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open browser after generation")
    args = parser.parse_args()

    path = run_full_report(
        n_formula=args.n_formula,
        n_book=args.n_book,
        n_nn=args.n_nn,
        output=args.output,
    )

    if not args.no_browser:
        webbrowser.open(f"file://{path}")

if __name__ == "__main__":
    main()
