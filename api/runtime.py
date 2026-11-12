"""Active dataset and active model: switching, model status, errors.

One ``Runtime`` per app. The active ``Services`` stays on ``app.state.services`` (everything existing
keeps using it); a dataset switch builds the new services first, swaps, then closes the old DuckDB file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from api.deps import Services
from config.settings import Settings
from config.registry import (MODELS, REGISTRY, DatasetSpec, ModelSpec, dataset_info, dataset_status,
                             resolve_checkpoint, settings_for_dataset)
from data_engine.ingestion import IngestionEngine
from data_engine.storage import get_storage, release_storage
from models.engine import EngineUnavailable, engine_status, get_engine
from models.pipeline import EWSPipeline
from utils.logging import logger
from utils.timefmt import now_ist, to_ist

CLOSE_GRACE_S = 2.0  # lets in-flight light reads (alert lists, counts) finish before the old file closes


def err(status: int, code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, **extra})


class Runtime:
    def __init__(self, app: FastAPI, registry: dict[str, DatasetSpec] | None = None,
                 models: dict[str, ModelSpec] | None = None, close_grace: float = CLOSE_GRACE_S):
        self.app, self.registry = app, registry if registry is not None else REGISTRY
        self.models = models if models is not None else MODELS
        self.close_grace = close_grace
        self.active_model = "stformer"
        self.training_ckpt: str | None = None  # checkpoint of the latest console training run (progress file lives next to it)
        self.base: Settings | None = None  # the process settings: every dataset derives from them, never from another dataset
        self._switch = threading.Lock()
        self._loaded_at: dict[tuple[str, str], str] = {}
        self._errors: dict[tuple[str, str], str] = {}
        self._sha: dict[tuple[str, float], str] = {}

    # ------------------------------------------------------------------------------------- lookups
    @property
    def svc(self) -> Services:
        return self.app.state.services

    @property
    def dataset_id(self) -> str | None:
        return self.svc.dataset_id

    def dataset(self, dataset_id: str) -> DatasetSpec:
        if dataset_id not in self.registry:
            raise err(404, "unknown_dataset", f"unknown dataset '{dataset_id}'", known=list(self.registry))
        return self.registry[dataset_id]

    def model(self, model_id: str) -> ModelSpec:
        if model_id not in self.models:
            raise err(404, "unknown_model", f"unknown model '{model_id}'", known=list(self.models))
        return self.models[model_id]

    def require_active(self, dataset_id: str | None) -> None:
        if dataset_id is not None and dataset_id != self.dataset_id:
            raise err(409, "dataset_not_active", f"dataset '{dataset_id}' is not the active dataset "
                      f"('{self.dataset_id}'); select it first", active=self.dataset_id)

    def check_supported(self, model_id: str, dataset_id: str | None) -> ModelSpec:
        m = self.model(model_id)
        ds = self.registry.get(dataset_id or "")
        if ds is not None and model_id not in ds.supported_models:
            raise err(422, "model_unsupported_for_dataset", f"{m.display_name} is not supported on {ds.display_name}")
        return m

    # ------------------------------------------------------------------------------------ datasets
    def datasets(self) -> dict[str, Any]:
        return {"active": self.dataset_id, "datasets": [dataset_info(s, self.dataset_id) for s in self.registry.values()]}

    def select_dataset(self, dataset_id: str) -> dict[str, Any]:
        spec = self.dataset(dataset_id)
        status = dataset_status(spec)
        if not status["available"]:
            raise err(422, "dataset_unavailable", status["reason"], state=status["state"])
        if not self._switch.acquire(blocking=False):
            raise err(409, "busy", "another dataset switch is in progress")
        try:
            old = self.svc
            reset = False
            if old.dataset_id != dataset_id:
                active = old.jobs.active()
                if active:
                    raise err(409, "busy", f"{len(active)} background job(s) are queued or running; "
                              "wait for them to finish before switching dataset", jobs=[j.id for j in active])
                if not old.retire():
                    raise err(409, "busy", "a scoring or forecast request is running; retry in a moment")
                try:
                    new = self._build(old, spec)
                except Exception as exc:
                    old.retired = False  # the old dataset keeps serving
                    raise err(503, "dataset_unavailable", f"could not open {spec.display_name}: {type(exc).__name__}: {exc}") from exc
                self.app.state.services = new
                threading.Timer(self.close_grace, self._close_if_idle, args=(old.settings,)).start()
                logger.info("dataset switched: {} -> {}", old.dataset_id, dataset_id)
            if self.active_model not in spec.supported_models:
                self.active_model, reset = spec.default_model, True
            return {"active": dataset_id, "dataset": dataset_info(spec, dataset_id),
                    "model": {"active": self.active_model, "reset": reset,
                              "reason": f"previous model is not supported on {spec.display_name}" if reset else None},
                    "job": None}
        finally:
            self._switch.release()

    def _close_if_idle(self, old_settings: Settings) -> None:
        """Close a retired database after the grace period, unless it became active again meanwhile
        (A -> B -> A reuses A's cached connection: closing it by path would break the live service)."""
        with self._switch:
            cur = self.svc.settings.storage
            if cur.backend == old_settings.storage.backend and cur.duckdb_path == old_settings.storage.duckdb_path \
                    and cur.postgres_dsn == old_settings.storage.postgres_dsn:
                return
            release_storage(old_settings)

    def _build(self, old: Services, spec: DatasetSpec) -> Services:
        settings = settings_for_dataset(self.base or old.settings, spec)
        storage = get_storage(settings)
        pipeline = EWSPipeline(settings, storage=storage)
        if pipeline.cache.load() is None:
            release_storage(settings)
            raise RuntimeError("feature cache is empty")
        return Services(settings=settings, pipeline=pipeline,
                        ingestion=IngestionEngine(storage, batch_rows=settings.storage.stream_batch_rows),
                        broker=old.broker, jobs=old.jobs, dataset_id=spec.id)

    # --------------------------------------------------------------------------------------- models
    def _settings(self, dataset_id: str | None):
        spec = self.registry.get(dataset_id or "")
        return settings_for_dataset(self.base or self.svc.settings, spec) if spec else self.svc.settings

    def _checkpoint_sha(self, path: Path) -> str | None:
        try:
            key = (str(path), path.stat().st_mtime)
        except OSError:
            return None
        if key not in self._sha:
            self._sha[key] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return self._sha[key]

    def model_info(self, model_id: str, dataset_id: str | None = None) -> dict[str, Any]:
        m = self.model(model_id)
        dataset_id = dataset_id or self.dataset_id
        ds = self.registry.get(dataset_id or "")
        settings = self._settings(dataset_id)
        st = engine_status(settings)[m.backend]
        ck = Path(settings.model.checkpoint_path)
        key = (m.id, str(settings.model.timesfm_checkpoint) if m.id == "timesfm3" else str(ck))
        compatible = ds is None or m.id in ds.supported_models
        loaded = bool(st["loaded"])
        if loaded:
            self._loaded_at.setdefault(key, now_ist())  # lazily loaded by /forecast: first seen loaded
            state, message = "loaded", "Loaded and ready."
        elif m.id == "stformer" and not ck.exists():
            state, message = "missing_checkpoint", f"No trained checkpoint for this dataset yet ({ck.name}); train one in the Model Lab."
        elif m.id == "timesfm3" and not settings.model.allow_noncommercial_weights:
            state, message = "licence_not_enabled", ("Disabled: the weights are non-commercial. Set "
                                                      "EWS_MODEL__ALLOW_NONCOMMERCIAL_WEIGHTS=true in .local/env.sh for research use.")
        elif m.id == "timesfm3" and importlib.util.find_spec("timesfm3") is None:
            state, message = "package_missing", "The timesfm3 package is not installed."
        elif m.id == "timesfm3" and not st.get("weights_cached"):
            state, message = "weights_not_downloaded", "Pretrained weights are not downloaded (1.3 GB); select with allow_download."
        elif key in self._errors:
            state, message = "error", self._errors[key]
        else:
            state, message = "ready", "Ready to load."
        version = st["model_version"] if loaded else (f"stformer:{self._checkpoint_sha(ck)}" if m.id == "stformer" and ck.exists() else None)
        device = None
        if loaded:
            try:
                d = get_engine(settings, m.backend).describe().get("device")
                device = d.get("device") if isinstance(d, dict) else d
            except EngineUnavailable:
                pass
        return {
            "id": m.id, "display_name": m.display_name, "backend": m.backend, "kind": m.kind, "description": m.description,
            "compatible": compatible,
            "incompatibility_reason": None if compatible else f"{m.display_name} is not supported on {ds.display_name}.",
            "supports_training": m.supports_training, "supports_quantiles": m.supports_quantiles,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "horizons": list(m.horizons),
            "targets": [t.id for t in (ds.targets if ds else ()) if t.channel in m.targets and t.available],
            "licence": {"name": m.licence_name, "restriction": m.licence_restriction, "badge": m.badge},
            "status": {"state": state, "message": message, "loaded": loaded, "device": device,
                       "loaded_at": self._loaded_at.get(key) if loaded else None,
                       "error": self._errors.get(key)},
            "version": version,
            "training": self._training(ck) if m.supports_training else None,
            "zero_shot": None if m.supports_training else {
                "checkpoint": m.checkpoint, "weights_cached": bool(st.get("weights_cached")), "size_gb": 1.32},
        }

    @staticmethod
    def _training(ck: Path) -> dict[str, Any] | None:
        out: dict[str, Any] = {}
        for p in (ck.with_suffix(".json"), ck.with_suffix(".progress.json")):
            try:
                out.update(json.loads(p.read_text()))
            except (OSError, ValueError):
                pass
        if not out:
            return None
        prog = out.get("status")
        return {**{k: out.get(k) for k in ("epochs", "best_epoch", "train_windows", "val_windows", "best_val_pinball",
                                           "val_coverage_10_90", "val_median_mape", "checkpoint", "device")},
                "trained_at": to_ist(out.get("started_at"), naive="ist"), "progress_status": prog}

    def models_list(self, dataset_id: str | None = None) -> dict[str, Any]:
        dataset_id = dataset_id or self.dataset_id
        if dataset_id is not None:
            self.dataset(dataset_id)  # 404 when unknown
        return {"dataset_id": dataset_id, "active": self.active_model,
                "models": [self.model_info(m, dataset_id) for m in self.models]}

    def select_model(self, model_id: str, allow_download: bool = False) -> dict[str, Any]:
        self.check_supported(model_id, self.dataset_id)
        m, settings = self.models[model_id], self.svc.settings
        info = self.model_info(model_id)
        if info["status"]["state"] in ("missing_checkpoint", "licence_not_enabled", "package_missing"):
            raise err(503, "model_unavailable", info["status"]["message"], state=info["status"]["state"])
        if info["status"]["state"] == "weights_not_downloaded" and not allow_download:
            raise err(503, "model_unavailable", info["status"]["message"], state="weights_not_downloaded")
        key = (m.id, str(settings.model.timesfm_checkpoint) if m.id == "timesfm3" else str(settings.model.checkpoint_path))
        try:
            get_engine(settings, m.backend)  # loads once; a failure leaves the previous model active
        except EngineUnavailable as exc:
            self._errors[key] = str(exc)
            raise err(503, "model_unavailable", str(exc)) from exc
        except Exception as exc:
            self._errors[key] = f"{type(exc).__name__}: {exc}"
            raise err(503, "model_load_failed", self._errors[key]) from exc
        self._errors.pop(key, None)
        self._loaded_at.setdefault(key, now_ist())
        self.active_model = m.id
        return {"active": m.id, "model": self.model_info(m.id)}

    def engine_reloaded(self) -> None:
        """A training run replaced the checkpoint: forget load times so the next status shows a fresh load."""
        self._loaded_at = {k: v for k, v in self._loaded_at.items() if k[0] != "stformer"}
