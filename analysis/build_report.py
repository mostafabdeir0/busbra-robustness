"""Assemble the final report: fill table placeholders, render HTML, print a compact PDF.

    python analysis/build_report.py --dir reports/submission_v2

Needs the tables written by build_report_tables.py and the figures in <dir>/figures.
PDF rendering uses a locally installed Microsoft Edge or Google Chrome in headless mode (optional).
"""
import argparse
import base64
import re
import subprocess
from pathlib import Path

import markdown

TABLES = {
    "chosen_hyperparameters": "table_chosen_hyperparameters.md",
    "fold_variation": "table_fold_variation.md",
    "original_eight_conditions": "table_original_eight_conditions.md",
    "coherent_development": "table_coherent_development.md",
    "stability_frequency": "table_stability_selection_frequency.md",
    "failure_breakdown": "table_failure_breakdown.md",
    "complexity": "table_complexity.md",
    "threshold_fixed": "table_threshold_fixed.md",
    "threshold_policy": "table_threshold_policy.md",
    "lr_coefficients": "table_lr_coefficients.md",
}
AMERICAN = {"characterise": "characterize", "normalisation": "normalization", "quantisation": "quantization",
            "regularised": "regularized", "favourable": "favorable", "behaviour": "behavior",
            "generalisation": "generalization", "neighbour": "neighbor", "standardiser": "standardizer",
            "labelled": "labeled", "Regularised": "Regularized"}
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.2pt; line-height: 1.38; color: #1b2733; max-width: 100%; }
h1 { font-size: 20pt; margin: 0 0 4px; color: #14304a; }
h2 { font-size: 14pt; margin: 20px 0 6px; color: #14527a; border-bottom: 1px solid #c9d6e2; padding-bottom: 2px; }
h3 { font-size: 11.5pt; margin: 14px 0 4px; color: #14527a; }
p { margin: 5px 0 7px; }
h1, h2, h3, h4 { break-after: avoid; page-break-after: avoid; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
table { border-collapse: collapse; margin: 8px 0 10px; font-size: 8.8pt; width: 100%; }
th, td { border: 1px solid #c9d6e2; padding: 3px 6px; text-align: left; vertical-align: top; }
th { background: #eaf1f7; }
img { max-width: 78%; display: block; margin: 8px auto 2px; page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f1f4f7; padding: 0 2px; }
a { color: #14527a; }
table:last-of-type, table.keep { break-inside: avoid; page-break-inside: avoid; }
"""


def find_browser():
    for path in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if Path(path).is_file():
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("reports/submission_v2"))
    args = parser.parse_args()
    folder = args.dir.resolve()
    text = (folder / "BUS_BRA_Report_template.md").read_text(encoding="utf-8")
    for key, filename in TABLES.items():
        table = (folder / "tables" / filename).read_text(encoding="utf-8").strip()
        token = "{{T:" + key + "}}"
        if token not in text:
            raise SystemExit(f"Placeholder missing in template: {token}")
        text = text.replace(token, table)
    if "{{" in text:
        raise SystemExit("Unfilled placeholder remains: " + re.findall(r"\{\{[^}]*\}\}", text)[0])
    counter = iter(range(1, 1000))   # number table captions in order of appearance
    text = re.sub(r"(?m)^Table: (.*)$", lambda m: f"*Table {next(counter)}.* {m.group(1)}", text)
    for british, american in AMERICAN.items():
        text = text.replace(british, american)
    (folder / "BUS_BRA_Report.md").write_text(text, encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])

    def keep_small(match):  # short tables (10 body rows or fewer, little text) stay on one page; long ones may split
        table = match.group(0)
        rows = table.count("<tr>") - 1
        chars = len(re.sub(r"<[^>]+>", "", table))
        return table.replace("<table>", '<table class="keep">', 1) if rows <= 10 and chars <= 1000 else table

    body = re.sub(r"<table>.*?</table>", keep_small, body, flags=re.S)

    def embed(match):  # embed figures so the PDF never races image loading
        image = folder / match.group(1)
        data = base64.b64encode(image.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    body = re.sub(r'src="(figures/[^"]+\.png)"', embed, body)
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>BUS-BRA final report</title><style>{CSS}</style></head><body>{body}</body></html>"
    (folder / "BUS_BRA_Report.html").write_text(html, encoding="utf-8")
    browser = find_browser()
    if browser is None:
        print("No Edge/Chrome found; HTML written, print it to PDF manually.")
        return
    pdf = folder / "BUS_BRA_Report.pdf"
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", (folder / "BUS_BRA_Report.html").as_uri()],
                   check=True, timeout=180)
    print("PDF:", pdf, round(pdf.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
