# tests/test_report.py

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from skipper_ai.report import build_report


class TestReport(unittest.TestCase):
    def test_build_report_embeds_charts_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            df = pd.DataFrame(
                {
                    "timestamp": [1000, 1001, 1002],
                    "boat_speed": [8.0, 7.5, 8.2],
                    "expected_speed": [9.0, 9.0, 9.0],
                    "performance_ratio": [0.89, 0.83, 0.91],
                }
            )
            df.to_csv(run_dir / "data.csv", index=False)

            analysis = {
                "run_id": "fixture_run",
                "avg_performance": 0.877,
                "avg_expected_speed": 9.0,
                "avg_boat_speed": 7.9,
                "feature_importance": {
                    "tws": 0.02,
                    "twa": 0.05,
                    "heel": 0.12,
                    "sail_id_numeric": 0.08,
                },
                "feature_impact": {
                    "tws": -0.01,
                    "twa": 0.0,
                    "heel": -0.04,
                    "sail_id_numeric": -0.02,
                },
                "total_rows": 3,
                "under_performance_impact": {},
            }
            (run_dir / "analysis.json").write_text(
                json.dumps(analysis), encoding="utf-8"
            )

            explanation = """### Summary line

- First issue
- Second issue

**Bold recommendation** here.
"""
            out = build_report(
                str(run_dir),
                explanation_text=explanation,
            )

            self.assertEqual(out.resolve(), (run_dir / "report.html").resolve())
            self.assertTrue(out.exists())
            html = out.read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64,", html)
            self.assertIn("<h3>Summary line</h3>", html)
            self.assertIn("<ul>", html)
            self.assertIn("<strong>Bold recommendation</strong>", html)

    def test_build_report_no_explain(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            df = pd.DataFrame(
                {
                    "timestamp": [1, 2],
                    "boat_speed": [8.0, 8.0],
                    "expected_speed": [9.0, 9.0],
                    "performance_ratio": [0.9, 0.9],
                }
            )
            df.to_csv(run_dir / "data.csv", index=False)
            analysis = {
                "run_id": "x",
                "avg_performance": 0.9,
                "avg_expected_speed": 9.0,
                "avg_boat_speed": 8.0,
                "feature_importance": {"heel": 0.1},
                "feature_impact": {"heel": -0.02},
                "total_rows": 2,
                "under_performance_impact": {},
            }
            (run_dir / "analysis.json").write_text(
                json.dumps(analysis), encoding="utf-8"
            )

            out = build_report(str(run_dir), no_explain=True)
            html = out.read_text(encoding="utf-8")
            self.assertIn("--no-explain", html)
            self.assertNotIn("article class=\"advice\"", html)


if __name__ == "__main__":
    unittest.main()
