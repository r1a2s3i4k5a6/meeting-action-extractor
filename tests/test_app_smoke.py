"""Smoke test for app.py WITHOUT streamlit/pandas installed: runs the whole script against fake modules.

It proves the script imports correctly, the upload -> extract -> table -> download flow runs without errors,
and the Evaluate tab runs. It does NOT prove the widgets render in a real browser (run `streamlit run app.py` for that)."""
import runpy
import sys
import types
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

APP = Path(__file__).resolve().parent.parent / "app.py"


class FakeDataFrame:
    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def to_csv(self, index=False):
        cols = list(self.rows[0]) if self.rows else []
        return "\n".join([",".join(cols)] + [",".join(str(r.get(c, "")) for c in cols) for r in self.rows])


class Ctx:
    """Stands in for st / st.sidebar / a column / a tab / an expander. Records what the app displays."""

    def __init__(self, log, sample="01_launch_readiness.txt"):
        self.log, self.sample = log, sample
        self.column_config = types.SimpleNamespace(SelectboxColumn=lambda **kw: ("select", kw))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def columns(self, n):
        return [Ctx(self.log, self.sample) for _ in range(n)]

    def tabs(self, names):
        return [Ctx(self.log, self.sample) for _ in names]

    def expander(self, *a, **k):
        return Ctx(self.log, self.sample)

    def selectbox(self, label, options, *a, **k):
        return self.sample if label.startswith("...or load") else options[0]

    def text_area(self, label, value="", **k):
        return value

    def text_input(self, *a, **k):
        return ""

    def date_input(self, *a, **k):
        return date(2026, 9, 14)

    def slider(self, label, lo, hi, value, *a, **k):
        return value

    def number_input(self, label, lo, hi, value, *a, **k):
        return value

    def file_uploader(self, *a, **k):
        return None

    def multiselect(self, *a, **k):
        return []

    def checkbox(self, *a, **k):
        return False

    def button(self, *a, **k):
        return True

    def data_editor(self, df, **k):
        self.log.setdefault("editor_rows", []).extend(df.rows)
        return df

    def dataframe(self, df, **k):
        self.log.setdefault("dataframes", []).append(df.rows)

    def download_button(self, label, data, *a, **k):
        self.log.setdefault("downloads", {})[label] = data

    def stop(self):
        raise AssertionError("st.stop() was called: the app hit an error path")

    def error(self, msg):
        raise AssertionError(f"app showed an error: {msg}")

    def __getattr__(self, name):  # title, caption, header, write, info, warning, subheader, set_page_config ...
        def fn(*a, **k):
            self.log.setdefault(name, []).append(a)
        return fn


def run_app(sample="01_launch_readiness.txt"):
    log = {}
    st = Ctx(log, sample)
    st.sidebar = Ctx(log, sample)
    pd = types.SimpleNamespace(DataFrame=FakeDataFrame)
    with mock.patch.dict(sys.modules, {"streamlit": st, "pandas": pd}):
        runpy.run_path(str(APP), run_name="__main__")
    return log


class AppSmokeTests(unittest.TestCase):
    def test_extract_and_evaluate_flow(self):
        log = run_app()
        owners = {r["owner"] for r in log["editor_rows"]}
        self.assertIn("Marcus", owners)
        self.assertTrue(all(r["status"] in ("open", "needs_review") for r in log["editor_rows"]))
        self.assertIn("Download JSON", log["downloads"])
        self.assertIn("Marcus", log["downloads"]["Download CSV (with your edits)"])
        # Evaluate tab produced a score table per dataset (dev, blind, heldout, synthetic)
        score_tables = [t for t in log["dataframes"] if t and "f1" in t[0]]
        self.assertGreaterEqual(len(score_tables), 4)

    def test_vtt_sample(self):
        log = run_app("04_incident_review.vtt")
        self.assertTrue(log["editor_rows"])


if __name__ == "__main__":
    unittest.main()
