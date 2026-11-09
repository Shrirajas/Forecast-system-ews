"""Central, environment-driven configuration for the EWS platform.

Every value can be overridden with an ``EWS_``-prefixed environment variable
(nested sections use ``__``), e.g. ``EWS_STORAGE__BACKEND=postgres`` or
``EWS_MODEL__D_MODEL=256``. A ``.env`` file in the working directory is also
read.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The 9 discrete quantile tracks produced by the forecasting engine.
QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: Target channels forecast by the model (order matters: index 0 is cash balance).
TARGET_FEATURES: tuple[str, ...] = ("eod_balance", "utilization")

#: Past-only covariates (observed up to the as-of date, unknown in the future).
PAST_COVARIATES: tuple[str, ...] = (
    "cibil_score",
    "outward_std_30d",
    "missed_salary_flag",
    "payment_return_flag",
)

#: Past-and-future covariates (known in advance: contracts and calendars).
FUTURE_COVARIATES: tuple[str, ...] = ("emi_due_amount", "holiday_flag", "salary_day_flag")


class StorageSettings(BaseModel):
    backend: Literal["duckdb", "postgres"] = "duckdb"
    duckdb_path: Path = PROJECT_ROOT / "data" / "ews.duckdb"
    #: DuckDB worker threads (0 = DuckDB default = all cores).
    duckdb_threads: int = 0
    #: Upper bound on DuckDB's buffer pool; DuckDB spills to disk beyond it.
    duckdb_memory_limit: str = "4GB"
    postgres_dsn: str = "postgresql+psycopg2://ews:ews@localhost:5432/ews"
    #: Rows per Arrow record batch when streaming in/out of the database.
    stream_batch_rows: int = 65_536


class CacheSettings(BaseModel):
    #: Directory holding the memory-mapped tensor cache.
    root: Path = PROJECT_ROOT / "data" / "vector_cache"


class ModelSettings(BaseModel):
    #: ``stformer`` = in-repo spatial-temporal transformer (trainable, offline): the production engine.
    #: ``timesfm3`` = pretrained google/timesfm-3.0-pytorch (needs Hugging Face access). Its weights are
    #: licensed for non-commercial, non-production use only: research baseline, not for production.
    backend: Literal["stformer", "timesfm3"] = "stformer"
    checkpoint_path: Path = PROJECT_ROOT / "artifacts" / "models" / "ews_stformer.pt"
    timesfm_checkpoint: str = "google/timesfm-3.0-pytorch"
    #: Series per TimesFM forward pass (the library's own default is 4).
    timesfm_batch_size: int = 64
    #: Run TimesFM under bf16/fp16 autocast on CUDA. Off until validated against fp32
    #: (wQL and q10-q90 coverage on real balances); large balances are where bf16 rounding shows.
    timesfm_autocast: bool = False
    #: Must be true to load TimesFM 3.0 weights in any environment: their licence forbids
    #: commercial and production use, including commercial demos.
    allow_noncommercial_weights: bool = False

    context_length: int = 512  # 512 daily tokens ~= 18 months of EOD history
    max_horizon: int = 90
    patch_length: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    epochs: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    window_stride: int = 7
    val_fraction: float = 0.15
    num_workers: int = 2
    seed: int = 7

    # Hardware / memory safety (RTX 4080 Super: 16 GB VRAM)
    amp_dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    cuda_memory_fraction: float = 0.92
    inference_batch_size: int = 512
    #: torch.compile the network (CUDA only) for training and inference.
    compile_model: bool = False

    @field_validator("context_length")
    @classmethod
    def _ctx_multiple(cls, v: int) -> int:
        if v % 32:
            raise ValueError("context_length must be a multiple of 32")
        return v


class RiskSettings(BaseModel):
    #: Quantile track that represents worst-case liability (cash) risk.
    liability_quantile: float = 0.1
    #: Quantile track that represents worst-case asset (credit-line) risk.
    asset_quantile: float = 0.9
    #: Utilisation level treated as a credit-line breach on the asset track.
    utilization_breach: float = 0.95
    #: Balance at/below this is treated as "cash reaches zero".
    zero_balance_threshold: float = 500.0
    #: Noisy-OR weights blending the three default drivers into one PD.
    util_pd_weight: float = 0.35
    bureau_pd_weight: float = 0.25
    #: Logistic bureau PD: PD = 1 / (1 + exp((CIBIL - midpoint) / scale)).
    bureau_midpoint: float = 600.0
    bureau_scale: float = 40.0
    #: Basel-style credit conversion factor for undrawn credit lines.
    undrawn_ccf: float = 0.75
    #: Days of urgency decay used to rank alerts (half weight at this many days).
    urgency_half_life_days: float = 30.0


class BehaviorSettings(BaseModel):
    #: Projected average-balance drop (vs trailing 90d average) that flags attrition.
    attrition_drop_threshold: float = 0.35
    #: Ignore attrition on accounts whose trailing average balance is below this.
    attrition_min_balance: float = 25_000.0
    #: Annual spread between replacement funding and CASA rate (cost of funds).
    replacement_funding_spread: float = 0.035


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EWS_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "Banking EWS & Loan Loss Predictor"
    environment: Literal["sandbox", "dev", "prod"] = "sandbox"
    log_level: str = "INFO"
    data_dir: Path = PROJECT_ROOT / "data"
    sandbox_dir: Path = PROJECT_ROOT / "data" / "sandbox"
    upload_spool_dir: Path = PROJECT_ROOT / "data" / "spool"
    #: Thread pool size for ingestion / aggregation / inference work.
    worker_threads: int = 4
    #: Build the tensor cache automatically when the API boots (if data exists).
    warm_cache_on_startup: bool = True
    #: Score the whole book after every refresh (and at startup when no stored run
    #: matches the current data), so whole-book reads never run the model.
    rescore_after_refresh: bool = True
    #: Horizons stored for whole-book reads; one forward pass serves all of them.
    book_horizons: tuple[int, ...] = (30, 60, 90)
    #: Accounts per shard when scoring the whole book (bounds memory on large books).
    book_shard_size: int = 50_000
    #: The one book horizon that raises alerts; the others only store scores.
    book_alert_horizon: int = 90
    #: Stored book runs kept per horizon; older runs' per-account scores are deleted.
    keep_book_runs: int = 3

    storage: StorageSettings = Field(default_factory=StorageSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    behavior: BehaviorSettings = Field(default_factory=BehaviorSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
