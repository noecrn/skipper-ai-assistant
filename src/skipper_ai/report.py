# src/skipper_ai/report.py

import base64
import io
import json
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FEATURE_LABELS = {
    "tws": "True wind speed (TWS)",
    "twa": "True wind angle (TWA)",
    "heel": "Heel angle",
    "sail_id_numeric": "Sail mode",
}

UNDER_POLAR_THRESHOLD = 0.95

REPORT_CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; line-height: 1.5; max-width: 56rem; margin: 0 auto; padding: 1.5rem; }
h1 { font-size: 1.75rem; margin-top: 0; }
h2 { font-size: 1.25rem; margin-top: 2rem; border-bottom: 1px solid #ccc; padding-bottom: 0.25rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #8884; padding: 0.5rem 0.75rem; text-align: left; }
th { background: #0001; }
figure { margin: 1.5rem 0; }
figure img { max-width: 100%; height: auto; border: 1px solid #8884; border-radius: 4px; }
article.advice { margin-top: 2rem; padding: 1rem 1.25rem; background: #00000008; border-radius: 8px; }
article.advice h3 { margin-top: 1.25rem; }
article.advice h3:first-child { margin-top: 0; }
.meta { color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }
"""


def _fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.standard_b64encode(buf.read()).decode("ascii")


def _chart_performance_ratio(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = df["timestamp"] if "timestamp" in df.columns else df.index
    ax.plot(x, df["performance_ratio"], color="steelblue", linewidth=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Polar (1.0)")
    ax.axhline(UNDER_POLAR_THRESHOLD, color="coral", linestyle=":", linewidth=1, alpha=0.8)
    ax.set_xlabel("Time (timestamp)" if "timestamp" in df.columns else "Sample index")
    ax.set_ylabel("Performance ratio")
    ax.set_title("Performance ratio over run")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _fig_to_base64_png(fig)


def _chart_boat_vs_expected(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = df["timestamp"] if "timestamp" in df.columns else df.index
    if "boat_speed" in df.columns and "expected_speed" in df.columns:
        ax.plot(x, df["boat_speed"], label="Boat speed", color="navy", linewidth=0.9)
        ax.plot(x, df["expected_speed"], label="Expected (polar)", color="darkorange", linewidth=0.9, alpha=0.85)
    ax.set_xlabel("Time (timestamp)" if "timestamp" in df.columns else "Sample index")
    ax.set_ylabel("Speed (knots)")
    ax.set_title("Boat speed vs polar expectation")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _fig_to_base64_png(fig)


def _chart_feature_bars(importance: dict, impact: dict) -> str:
    keys = list(importance.keys())
    labels = [FEATURE_LABELS.get(k, k) for k in keys]
    imp_vals = [importance[k] for k in keys]
    impact_vals = [impact.get(k, 0.0) for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    y_pos = range(len(labels))
    ax1.barh(list(y_pos), imp_vals, color="steelblue")
    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Mean |contribution|")
    ax1.set_title("Factor influence (magnitude)")
    ax1.grid(True, axis="x", alpha=0.3)

    colors = ["#c44" if v < 0 else "#4a4" for v in impact_vals]
    ax2.barh(list(y_pos), impact_vals, color=colors)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("Average signed effect")
    ax2.set_title("Average push on performance")
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64_png(fig)


def _sorted_feature_rows(importance: dict, impact: dict) -> list[tuple[str, float, float]]:
    rows = []
    for k in sorted(importance.keys(), key=lambda x: importance[x], reverse=True):
        label = FEATURE_LABELS.get(k, k)
        rows.append((label, float(importance[k]), float(impact.get(k, 0.0))))
    return rows


def _pct_under_polar(df: pd.DataFrame) -> float:
    if df.empty or "performance_ratio" not in df.columns:
        return 0.0
    pr = df["performance_ratio"]
    return float((pr < UNDER_POLAR_THRESHOLD).mean() * 100.0)


def build_report(
    run_dir: str,
    *,
    no_explain: bool = False,
    explanation_text: str | None = None,
) -> Path:
    """
    Build a single self-contained report.html under run_dir.
    Loads data.csv and analysis.json from run_dir.
    If explanation_text is None and no_explain is False, loads explanation.txt
    if present; otherwise leaves advice section empty unless caller passes text.
    """
    run_path = Path(run_dir).resolve()
    data_path = run_path / "data.csv"
    analysis_path = run_path / "analysis.json"
    explanation_path = run_path / "explanation.txt"
    out_path = run_path / "report.html"

    if not data_path.exists():
        raise FileNotFoundError(f"Run data not found: {data_path}")
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis not found: {analysis_path}")

    df = pd.read_csv(data_path)
    with open(analysis_path, encoding="utf-8") as f:
        analysis = json.load(f)

    run_id = analysis.get("run_id", run_path.name)
    total_rows = int(analysis.get("total_rows", len(df)))
    avg_perf = float(analysis.get("avg_performance", 0.0))
    avg_boat = float(analysis.get("avg_boat_speed", 0.0))
    avg_exp = float(analysis.get("avg_expected_speed", 0.0))
    importance = analysis.get("feature_importance", {})
    impact = analysis.get("feature_impact", {})

    pct_under = _pct_under_polar(df)
    feature_rows = _sorted_feature_rows(importance, impact)

    img_ratio = _chart_performance_ratio(df)
    img_speed = _chart_boat_vs_expected(df)
    img_features = _chart_feature_bars(importance, impact)

    advice_html = ""
    if not no_explain:
        text = explanation_text
        if text is None and explanation_path.exists():
            text = explanation_path.read_text(encoding="utf-8")
        if text:
            advice_html = markdown.markdown(text, extensions=["extra"])

    feature_table_rows = "".join(
        f"<tr><td>{label}</td><td>{imp:.6f}</td><td>{impct:+.6f}</td></tr>"
        for label, imp, impct in feature_rows
    )

    advice_block = ""
    if no_explain:
        advice_block = "<p><em>Coach advice omitted (--no-explain).</em></p>"
    elif advice_html:
        advice_block = f'<article class="advice">{advice_html}</article>'
    else:
        advice_block = "<p><em>No explanation text available. Run explain or pass explanation when building the report.</em></p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Skipper AI — Run report: {run_id}</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
  <h1>Performance report: {run_id}</h1>
  <p class="meta">
    Rows analyzed: {total_rows} · Avg performance ratio: {avg_perf:.3f}
    ({avg_perf * 100:.1f}% of polar) · Avg boat / expected speed: {avg_boat:.2f} / {avg_exp:.2f} kts
    · Samples below {UNDER_POLAR_THRESHOLD:.2f} polar ratio: {pct_under:.1f}%
  </p>

  <h2>Key metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Average performance ratio</td><td>{avg_perf:.4f}</td></tr>
      <tr><td>% samples under {UNDER_POLAR_THRESHOLD} ratio</td><td>{pct_under:.1f}%</td></tr>
      <tr><td>Average boat speed</td><td>{avg_boat:.2f} kts</td></tr>
      <tr><td>Average expected (polar) speed</td><td>{avg_exp:.2f} kts</td></tr>
    </tbody>
  </table>

  <h2>Factor contributions</h2>
  <table>
    <thead><tr><th>Factor</th><th>Influence (magnitude)</th><th>Average signed effect</th></tr></thead>
    <tbody>{feature_table_rows}</tbody>
  </table>

  <h2>Charts</h2>
  <figure><img alt="Performance ratio over time" src="data:image/png;base64,{img_ratio}" /></figure>
  <figure><img alt="Boat vs expected speed" src="data:image/png;base64,{img_speed}" /></figure>
  <figure><img alt="Feature influence" src="data:image/png;base64,{img_features}" /></figure>

  <h2>Coach advice</h2>
  {advice_block}
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    return out_path
