"""Saved model comparisons: ``artifacts/eval/<dataset>/<comparison-id>.json`` (written by the ML comparison code)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from api.runtime import err
from config.registry import DatasetSpec
from utils.timefmt import to_ist

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,95}$")


def eval_dir(spec: DatasetSpec) -> Path:
    return spec.checkpoint_path.parent.parent / "eval" / spec.id


def _headline(doc: dict[str, Any]) -> dict[str, Any]:
    """{model: {mae, wape}} from whichever aggregate block the comparison file carries."""
    by = doc.get("aggregate") or (doc.get("metrics") or {}).get("by_model") or {}
    return {m: {k: (v.get("metrics") or v).get(k) for k in ("mae", "wape")} for m, v in by.items()}


def summary(doc: dict[str, Any], spec: DatasetSpec) -> dict[str, Any]:
    models = doc.get("models")
    scope = doc.get("scope")
    return {"comparison_id": doc.get("comparison_id") or doc.get("id"), "created_at": to_ist(doc.get("created_at"), naive="ist"),
            "dataset_id": spec.id, "scope": scope.get("accounts") if isinstance(scope, dict) else scope,
            "horizon": doc.get("horizon"), "models": list(models) if isinstance(models, dict) else models,
            "origins": doc.get("origins"), "headline": _headline(doc)}


def list_saved(spec: DatasetSpec) -> list[dict[str, Any]]:
    out = []
    for p in eval_dir(spec).glob("*.json"):
        try:
            out.append(summary(json.loads(p.read_text()), spec))
        except (OSError, ValueError):
            continue  # a half-written or foreign file is not a comparison
    return sorted(out, key=lambda d: d["created_at"] or "", reverse=True)


def load(spec: DatasetSpec, comparison_id: str) -> dict[str, Any]:
    if not ID_RE.match(comparison_id):  # also blocks path traversal
        raise err(404, "not_found", "comparison not found")
    p = eval_dir(spec) / f"{comparison_id}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        raise err(404, "not_found", f"comparison '{comparison_id}' not found for dataset '{spec.id}'") from None
