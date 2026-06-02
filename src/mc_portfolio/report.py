"""Simple HTML report export for simulation results."""
from __future__ import annotations

import html
import json
import os
from typing import Dict

from .config import OUTPUT_DIR


def write_html_report(result: Dict, filename: str = "report.html") -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    meta = result.get("metadata", {})
    summary = result.get("summary", {})
    optimization = result.get("optimization")
    stress = result.get("stress_tests")

    cards = "".join(
        f"<div class='card'><b>{html.escape(str(k))}</b><br>{html.escape(str(round(v, 4) if isinstance(v, float) else v))}</div>"
        for k, v in summary.items()
        if not isinstance(v, dict)
    )
    plots = "".join(f"<img src='{html.escape(os.path.basename(p))}' alt='plot'>" for p in meta.get("plots", []))
    opt_html = f"<h2>Optimization</h2><pre>{html.escape(json.dumps(optimization, indent=2)[:6000])}</pre>" if optimization else ""
    stress_html = f"<h2>Stress Tests</h2><pre>{html.escape(json.dumps(stress, indent=2)[:6000])}</pre>" if stress else ""

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>Monte Carlo Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;line-height:1.45;color:#17202a}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card{{border:1px solid #d8dee9;border-radius:12px;padding:12px;background:#f8fafc}}
img{{max-width:100%;border:1px solid #d8dee9;border-radius:12px;margin:14px 0}}
pre{{white-space:pre-wrap;background:#f8fafc;border:1px solid #d8dee9;border-radius:12px;padding:16px}}
</style></head><body>
<h1>Monte Carlo Portfolio Simulator Report</h1>
<h2>Metadata</h2><pre>{html.escape(json.dumps(meta, indent=2))}</pre>
<h2>Risk Summary</h2><div class='grid'>{cards}</div>
<h2>Charts</h2>{plots}
{opt_html}
{stress_html}
</body></html>
""")
    return path
