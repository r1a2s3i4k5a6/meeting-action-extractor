"""Evaluation: precision / recall / F1 for tasks, plus owner and deadline accuracy."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pipeline import run_pipeline
from .schema import ActionItem
from .similarity import similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MATCH_THRESHOLD = 0.5


def match_items(pred: Sequence[ActionItem], gold: Sequence[Dict[str, Any]],
                threshold: float = MATCH_THRESHOLD) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching of predicted to gold tasks by text similarity."""
    scored = sorted(((similarity(p.task, g["task"]), i, j)
                     for i, p in enumerate(pred) for j, g in enumerate(gold)), reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for s, i, j in scored:
        if s < threshold:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            pairs.append((i, j, s))
    return pairs


def same_owner(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return not a and not b
    return a.split()[0].lower() == b.split()[0].lower()


def score(per_meeting: Sequence[Tuple[List[ActionItem], List[Dict[str, Any]]]],
          min_conf: float = 0.0) -> Dict[str, Any]:
    tp = fp = fn = owner_ok = date_ok = 0
    tp_conf: List[float] = []
    fp_conf: List[float] = []
    for pred_all, gold in per_meeting:
        pred = [p for p in pred_all if p.confidence >= min_conf]
        pairs = match_items(pred, gold)
        matched_p = {i for i, _, _ in pairs}
        tp += len(pairs)
        fp += len(pred) - len(pairs)
        fn += len(gold) - len(pairs)
        for i, j, _ in pairs:
            owner_ok += same_owner(pred[i].owner, gold[j].get("owner"))
            date_ok += (pred[i].deadline == gold[j].get("deadline"))
            tp_conf.append(pred[i].confidence)
        fp_conf += [p.confidence for k, p in enumerate(pred) if k not in matched_p]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {"min_confidence": min_conf, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "owner_accuracy": round(owner_ok / tp, 3) if tp else 0.0,
            "deadline_accuracy": round(date_ok / tp, 3) if tp else 0.0,
            "avg_confidence_tp": avg(tp_conf), "avg_confidence_fp": avg(fp_conf)}


def breakdown(per_meeting, distractors, min_conf: float = 0.3) -> Dict[str, Any]:
    """Where does the extractor fail? Recall by sentence template / deadline kind, false positives by distractor kind.
    Needs gold items with 'template'/'when' keys and per-meeting 'distractors' (see scripts/generate_synthetic.py)."""
    by_tpl: Dict[str, List[int]] = {}
    by_when: Dict[str, List[int]] = {}
    fp_kind: Dict[str, int] = {}
    for (pred_all, gold), dis in zip(per_meeting, distractors):
        pred = [p for p in pred_all if p.confidence >= min_conf]
        pairs = match_items(pred, gold)
        hit_g = {j for _, j, _ in pairs}
        hit_p = {i for i, _, _ in pairs}
        for j, g in enumerate(gold):
            ok = j in hit_g and any(same_owner(pred[i].owner, g.get("owner")) and pred[i].deadline == g.get("deadline")
                                    for i, jj, _ in pairs if jj == j)
            for table, key in ((by_tpl, g.get("template")), (by_when, g.get("when"))):
                if key:
                    r = table.setdefault(key, [0, 0, 0])
                    r[0] += j in hit_g   # found
                    r[1] += ok           # found with correct owner AND deadline
                    r[2] += 1
        for i, p in enumerate(pred):
            if i in hit_p:
                continue
            body = p.source.split(":", 1)[-1].strip().lower()
            kind = "other"
            for d in dis:
                dt = d["text"].lower()
                if body and (body in dt or dt in body):
                    kind = d["kind"]
                    break
            fp_kind[kind] = fp_kind.get(kind, 0) + 1
    fmt = lambda t: {k: {"found": v[0], "fully_correct": v[1], "total": v[2]} for k, v in sorted(t.items())}
    return {"by_template": fmt(by_tpl), "by_deadline_kind": fmt(by_when), "false_positives_by_kind": dict(sorted(fp_kind.items(), key=lambda kv: -kv[1]))}


def format_breakdown(b: Dict[str, Any]) -> str:
    lines = ["", "Recall by sentence template (found / fully correct owner+deadline / total):"]
    for k, v in sorted(b["by_template"].items(), key=lambda kv: kv[1]["found"] / max(1, kv[1]["total"])):
        lines.append(f"  {k:<16} {v['found']:>3} / {v['fully_correct']:>3} / {v['total']:>3}   ({v['found'] / v['total']:.0%} found)")
    lines.append("Deadline kinds (found / fully correct / total):")
    for k, v in sorted(b["by_deadline_kind"].items(), key=lambda kv: kv[1]["fully_correct"] / max(1, kv[1]["total"])):
        lines.append(f"  {k:<18} {v['found']:>3} / {v['fully_correct']:>3} / {v['total']:>3}   ({v['fully_correct'] / v['total']:.0%} fully correct)")
    lines.append("False positives by what the sentence really was: " + ", ".join(f"{k}={v}" for k, v in b["false_positives_by_kind"].items()))
    return "\n".join(lines)


class BackendFallbackError(RuntimeError):
    """The requested backend could not run, so the pipeline silently used another one."""


def evaluate_dataset(data_dir: Path = DATA_DIR, backend: str = "rules",
                     thresholds: Sequence[float] = (0.0, 0.3, 0.5, 0.7), limit: int = 0,
                     strict: bool = True, **pipeline_kwargs) -> Dict[str, Any]:
    """Score `backend` on one annotated dataset.

    strict=True (default): if the pipeline fell back to a different backend (e.g. the LLM call failed and the rules
    ran instead) raise BackendFallbackError, so rules scores can never be reported under the label "llm"."""
    ann = json.loads((Path(data_dir) / "annotations.json").read_text(encoding="utf-8"))
    per_meeting, dis = [], []
    for m in (ann["meetings"][:limit] if limit else ann["meetings"]):
        text = (Path(data_dir) / "sample_transcripts" / m["file"]).read_text(encoding="utf-8")
        res = run_pipeline(text, date.fromisoformat(m["meeting_date"]), backend=backend,
                           min_confidence=0.0, **pipeline_kwargs)
        if strict and backend in ("rules", "llm") and res.backend != backend:
            raise BackendFallbackError(
                f"{m['file']}: requested backend '{backend}' but '{res.backend}' was used. "
                f"{' '.join(res.warnings) or 'No warning given.'}")
        per_meeting.append((res.items, m["action_items"]))
        dis.append(m.get("distractors", []))
    extra = {}
    if any(g.get("template") for _, gold in per_meeting for g in gold):
        extra["breakdown"] = breakdown(per_meeting, dis)
    return {**extra, "backend": backend, "n_meetings": len(per_meeting),
            "n_gold_items": sum(len(g) for _, g in per_meeting),
            "by_threshold": [score(per_meeting, t) for t in thresholds]}


def format_report(rep: Dict[str, Any]) -> str:
    lines = [f"Backend: {rep['backend']} | meetings: {rep['n_meetings']} | gold items: {rep['n_gold_items']}",
             f"{'min_conf':>8} {'P':>6} {'R':>6} {'F1':>6} {'owner':>6} {'date':>6} {'TP':>4} {'FP':>4} {'FN':>4}"]
    for r in rep["by_threshold"]:
        lines.append(f"{r['min_confidence']:>8.2f} {r['precision']:>6.2f} {r['recall']:>6.2f} {r['f1']:>6.2f} "
                     f"{r['owner_accuracy']:>6.2f} {r['deadline_accuracy']:>6.2f} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}")
    return "\n".join(lines)


def compare_backends(data_dir: Path, backends: Sequence[str] = ("rules", "llm"), limit: int = 0,
                     min_conf: float = 0.3, **pipeline_kwargs) -> Dict[str, Any]:
    """Run several backends on the SAME meetings and return their scores at `min_conf` side by side."""
    out: Dict[str, Any] = {"dataset": Path(data_dir).name, "min_confidence": min_conf, "backends": {}}
    for b in backends:
        rep = evaluate_dataset(Path(data_dir), b, thresholds=(min_conf,), limit=limit, **pipeline_kwargs)
        out["backends"][b] = {"n_meetings": rep["n_meetings"], "n_gold_items": rep["n_gold_items"], **rep["by_threshold"][0]}
    return out


def format_comparison(cmp: Dict[str, Any]) -> str:
    lines = [f"[{cmp['dataset']}] backends compared at confidence >= {cmp['min_confidence']}",
             f"{'backend':<8} {'meetings':>8} {'gold':>5} {'P':>6} {'R':>6} {'F1':>6} {'owner':>6} {'date':>6} {'TP':>4} {'FP':>4} {'FN':>4}"]
    for b, r in cmp["backends"].items():
        lines.append(f"{b:<8} {r['n_meetings']:>8} {r['n_gold_items']:>5} {r['precision']:>6.2f} {r['recall']:>6.2f} {r['f1']:>6.2f} "
                     f"{r['owner_accuracy']:>6.2f} {r['deadline_accuracy']:>6.2f} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate extraction accuracy on annotated meetings.")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--all", action="store_true", help="Evaluate every dataset under data/ (dev, heldout, blind)")
    ap.add_argument("--backend", default="rules", choices=["rules", "llm"])
    ap.add_argument("--limit", type=int, default=0, help="Only the first N meetings (saves LLM cost)")
    ap.add_argument("--compare", action="store_true",
                    help="Run rules AND llm on the same meetings and print them side by side "
                         "(needs ANTHROPIC_API_KEY; saved to reports/comparison.json)")
    ap.add_argument("--out", default=None, help="Output JSON (default reports/evaluation.json, or reports/comparison.json with --compare)")
    a = ap.parse_args(argv)
    if a.compare:
        out = Path(a.out or ROOT / "reports" / "comparison.json")
        dirs = ([DATA_DIR] + sorted(p.parent for p in DATA_DIR.glob("*/annotations.json"))) if a.all else [Path(a.data_dir)]
        results = {}
        for d in dirs:
            try:
                results[d.name] = compare_backends(d, limit=a.limit)
            except BackendFallbackError as e:
                raise SystemExit(f"LLM backend did not run, so there is nothing to compare:\n  {e}")
            print(format_comparison(results[d.name]) + "\n")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved {out}")
        return
    a.out = a.out or str(ROOT / "reports" / "evaluation.json")
    if a.all:
        reports = {}
        for d in [DATA_DIR] + sorted(p.parent for p in DATA_DIR.glob("*/annotations.json")):
            reports[d.name] = evaluate_dataset(d, a.backend, limit=a.limit)
            print(f"\n[{d.name}] " + format_report(reports[d.name]))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nSaved {a.out}")
        return
    rep = evaluate_dataset(Path(a.data_dir), a.backend, limit=a.limit)
    print(format_report(rep))
    if "breakdown" in rep:
        print(format_breakdown(rep["breakdown"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    try:
        main()
    except BackendFallbackError as e:
        raise SystemExit(f"Evaluation stopped: the requested backend did not run.\n  {e}")
