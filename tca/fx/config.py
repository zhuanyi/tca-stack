"""Typed access to ``config/fx_sim.yaml`` and ``config/brokers.yaml``.

Every calibrated number is read from configuration; nothing here carries a default for a
modelled quantity. Units: durations in the unit named by the field, notionals in USD.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Window(_Model):
    """A UTC time-of-day window ``start`` to ``end`` ("HH:MM") with a mixture weight."""

    start: str
    end: str
    weight: float = 1.0


class Period(_Model):
    start: date
    end: date


class Pairs(_Model):
    order: list[str]
    weights: dict[str, float]


class VolumeProxy(_Model):
    spot_usd_bn_per_day: dict[str, float]


class Notional(_Model):
    median: float
    sigma: float
    min: float
    max: float
    round_snap_probability: float
    round_snap_values_mm: list[float]


class Side(_Model):
    buy_probability_base: float
    daily_bias_sigma: float
    buy_probability_floor: float
    buy_probability_cap: float


class Poisson(_Model):
    mean: float


class Flow(_Model):
    parents_per_weekday: Poisson
    arrival_time_utc: list[Window]
    notional_usd: Notional
    side: Side
    urgency_probabilities: dict[str, float]
    high_share_risk_transfer: float


class DurationRule(_Model):
    notional_mm_per_minute: float
    min_minutes: float
    max_minutes: float


class TwapStyle(_Model):
    duration_rule: DurationRule
    slice_interval_s: tuple[float, float]
    passive_first_s: float


class PovStyle(_Model):
    participation: float
    max_duration_minutes: float
    slice_interval_s: float


class IsStyle(_Model):
    duration_rule: DurationRule
    risk_aversion: float
    slice_interval_s: float


class RiskTransferStyle(_Model):
    hs_multiple: float
    k_rt: float


class FixStyle(_Model):
    duration_minutes: float
    aggressive_child_interval_s: float
    arrival_lead_minutes: tuple[float, float]


class Styles(_Model):
    TWAP: TwapStyle
    POV: PovStyle
    IS: IsStyle
    RISK_TRANSFER: RiskTransferStyle
    FIX_ORDER: FixStyle


class BrokerAssignment(_Model):
    assignment_mix: dict[str, float]
    discretionary_large_threshold_usd: float
    brokers: list[str]


class Impact(_Model):
    y: float
    perm_frac: float
    half_life_minutes: float
    sigma_d_window_days: int
    passive_impact_multiple: float


class HoldMs(_Model):
    median_range: tuple[float, float]
    sigma: float


class LiquidityProviders(_Model):
    count: int
    markup_pips: tuple[float, float]
    reject_base: tuple[float, float]
    reject_asym: tuple[float, float]
    hold_ms: HoldMs
    max_clip_mm_by_lp_mm: list[float]
    quote_noise_pips: float


class LastLook(_Model):
    max_rejections_before_ecn: int
    ecn_markup_pips: float


class Latency(_Model):
    decision_latency_ms: float


class Passive(_Model):
    timeout_s: float
    target_fill_rate: float


class Simulation(_Model):
    seed: int
    ci_max_days: int
    n_traders: int
    markout_horizons_s: list[int]
    passive_calibration_samples: int
    passive_calibration_days: int


class Pretrade(_Model):
    t_ref_minutes: float
    duration_exponent: float
    duration_clip: tuple[float, float]
    style_factor: dict[str, float]


class Synthetic(_Model):
    start_level: dict[str, float]
    annual_vol: dict[str, float]
    base_spread_pips: dict[str, float]
    ticks_per_day: int
    hourly_spread_multiplier: list[float]
    hourly_tick_weight: list[float]


class FxConfig(_Model):
    """The parsed ``fx_sim.yaml``."""

    period: Period
    pairs: Pairs
    volume_proxy: VolumeProxy
    flow: Flow
    execution_styles: Styles
    broker_assignment: BrokerAssignment
    impact: Impact
    liquidity_providers: LiquidityProviders
    last_look: LastLook
    latency: Latency
    passive: Passive
    simulation: Simulation
    pretrade: Pretrade
    synthetic: Synthetic

    def daily_volume_usd(self, pair: str) -> float:
        """Daily spot-turnover proxy ``V`` of ``pair`` in USD."""
        return self.volume_proxy.spot_usd_bn_per_day[pair] * 1e9


class BrokerWindow(_Model):
    start: str
    end: str
    markup_bp: float


class BrokerCfg(_Model):
    """Planted effect of one broker; see ``config/brokers.yaml``."""

    markup_bp: float
    markup_windows: list[BrokerWindow] = Field(default_factory=list)
    impact_multiplier: float = 1.0
    post_fill_impact_bp: float = 0.0


def load_config(path: Path | None = None) -> FxConfig:
    """Load and validate ``fx_sim.yaml`` (default: ``config/fx_sim.yaml``)."""
    raw = yaml.safe_load(Path(path or CONFIG_DIR / "fx_sim.yaml").read_text())
    return FxConfig.model_validate(raw)


def load_brokers(path: Path | None = None) -> dict[str, BrokerCfg]:
    """Load ``brokers.yaml`` as a map from broker label to its planted effect."""
    raw = yaml.safe_load(Path(path or CONFIG_DIR / "brokers.yaml").read_text())
    return {name: BrokerCfg.model_validate(body) for name, body in raw["brokers"].items()}


def check_broker_sets(cfg: FxConfig, book: dict[str, BrokerCfg]) -> None:
    """Raise ``ValueError`` if ``brokers.yaml`` and ``fx_sim.yaml`` name different brokers."""
    used = set(cfg.broker_assignment.brokers)
    if used != set(book):
        raise ValueError(
            f"broker sets differ: brokers.yaml has {sorted(book)}, "
            f"fx_sim.yaml broker_assignment.brokers has {sorted(used)}"
        )


def hhmm_to_seconds(text: str) -> int:
    """Seconds after midnight of an ``"HH:MM"`` string."""
    hours, minutes = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60
