"""Typed view of ``config/ust_sim.yaml``; every calibrated number is read from the file."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "ust_sim.yaml"

TENORS = (2, 3, 5, 7, 10, 20, 30)
RUN_STATUSES = ("OTR", "OLD", "DOUBLE_OLD")


class Period(BaseModel):
    start: date
    end: date


class TimeGrid(BaseModel):
    tz: str
    start: str
    end: str
    step_s: int


class Bump(BaseModel):
    time: str | None = None
    offset_s_after_close: int | None = None
    height: float


class VarianceWeight(BaseModel):
    baseline_points: list[tuple[str, float]]
    after_16_00: float
    bump_halfwidth_s: float
    bump_data_release: Bump
    bump_auction_auctioned_tenor: Bump
    bump_auction_other_tenors: Bump
    bump_fomc: Bump
    bump_close_marks: Bump


class SecurityNoise(BaseModel):
    ou_half_life_min: float
    ou_stationary_sd_bp: float


class YieldProcess(BaseModel):
    variance_weight: VarianceWeight
    window_days: int
    n_factors: int
    security_noise: SecurityNoise
    off_the_run_yield_spread_bp: dict[str, float]


class TodMult(BaseModel):
    from_: str
    to: str
    mult: float


class Clip(BaseModel):
    spread: tuple[float, float]
    depth: tuple[float, float]
    impact: tuple[float, float]


class April2025(BaseModel):
    full_spread_32nds_peak: dict[int, float]
    depth_mm_trough: dict[int, float]
    price_impact_32nds_per_100mm_peak: dict[int, float]
    spec_vol_mult_peak: float


class CompositeQuote(BaseModel):
    hs_base_32nds: dict[int, float]
    hs_eps_sd: float
    hs_eps_resample_min: float
    run_mult: dict[str, float]
    tod_mult: list[TodMult]
    depth_mm: dict[int, float]
    depth_run_mult_exponent: float
    regime_reference_tenor: int
    regime_matched_on: date
    regime_clip: Clip
    april_2025_reference: April2025


class Impact(BaseModel):
    price_impact_32nds_per_100mm: dict[int, float]
    size_exponent: float
    perm_frac: float
    decay_half_life_min: float


class Leakage(BaseModel):
    p_hedge_lose: float
    phi_lose: float
    lose_hedge_window_min: tuple[float, float]
    win_hedge_window_min: tuple[float, float]
    hedge_clips: int


class Inventory(BaseModel):
    skew_coeff: float
    inventory_scale_mm: float
    ou_half_life_min: float
    ou_stationary_sd_scale: float


class QuoteRule(BaseModel):
    noise_sd_hs: float
    floor_half_spread_frac: float
    tick_denominator: int


class Dealers(BaseModel):
    n: int
    tier_counts: dict[int, int]
    tier_selection_weight: dict[int, float]
    spread_mult_range: dict[int, tuple[float, float]]
    axe_prob_range: tuple[float, float]
    axe_effect: float
    response_prob_base_range: tuple[float, float]
    response_latency_median_s_range: tuple[float, float]
    response_latency_sigma: float
    no_quote_size_mm_range: tuple[float, float]
    internalisation_range: tuple[float, float]
    size_charge_k_range: tuple[float, float]
    competition_widening_range: tuple[float, float]
    competition_free_dealers: int
    inventory: Inventory
    quote: QuoteRule


class Arrival(BaseModel):
    from_: str
    to: str
    weight: float


class SizeModel(BaseModel):
    median: dict[int, float]
    lognormal_sigma: float
    clip_mm: tuple[float, float]
    round_snap_prob: float
    round_snap_values: list[float]


class NDealersPolicy(BaseModel):
    otr_under_10mm: int
    otr_under_50mm: int
    otr_50_to_150mm: int
    otr_over_150mm: int
    off_the_run_over_25mm: int
    jitter: dict[str, float]
    single_dealer_axe_prob: float


class LifeCycle(BaseModel):
    min_award_wait_s: float
    pass_threshold_hs: float
    pass_slack_ticks: float


class OrderFlow(BaseModel):
    parents_per_day_poisson: float
    arrival_mixture: list[Arrival]
    no_order_before_auction_result_min: float
    send_delay_s: dict[str, float]
    tenor_weights: dict[int, float]
    run_status_probs: dict[str, float]
    size_mm: SizeModel
    side: dict[str, object]
    execution_policy_probs: dict[str, float]
    split: dict[str, list[float]]
    list_rfq: dict[str, list[int]]
    n_dealers_policy: NDealersPolicy
    rfq_life_cycle: LifeCycle
    traders: dict[str, int]


class TcaSettings(BaseModel):
    markout_horizons_s: list[int]
    face_usd_per_mm: float
    scorecard_tenor_buckets: dict[str, list[int]]


class SimConfig(BaseModel):
    """Simulator parameters; ``leakage_scale`` and ``competition_widening_scale`` are sweepable."""

    seed: int
    period: Period
    events_file: str
    leakage_scale: float
    competition_widening_scale: float
    ci_days: int
    time_grid: TimeGrid
    yield_process: YieldProcess
    composite_quote: CompositeQuote
    impact: Impact
    leakage: Leakage
    dealers: Dealers
    order_flow: OrderFlow
    tca: TcaSettings


def _rename_keys(node):
    """Map the YAML keys ``from`` and ``after_16:00`` onto valid Python attribute names."""
    if isinstance(node, dict):
        renames = {"from": "from_", "after_16:00": "after_16_00"}
        return {renames.get(k, k): _rename_keys(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_rename_keys(v) for v in node]
    return node


def _flatten(raw: dict) -> dict:
    """Lift the nested YAML sections into the flat shape of :class:`SimConfig`."""
    yp = raw["yield_process"]
    cq = raw["composite_quote"]
    return {
        **raw,
        "yield_process": {
            "variance_weight": yp["variance_weight"],
            "window_days": yp["regime"]["window_days"],
            "n_factors": yp["factors"]["n"],
            "security_noise": yp["security_noise"],
            "off_the_run_yield_spread_bp": yp["off_the_run_yield_spread_bp"],
        },
        "composite_quote": {
            **cq,
            "depth_run_mult_exponent": cq["depth_scaling"]["run_mult_exponent"],
            "regime_reference_tenor": cq["regime_channels"]["reference_tenor"],
            "regime_matched_on": cq["regime_channels"]["matched_on"],
            "regime_clip": cq["regime_channels"]["clip"],
        },
    }


def load_config(path: Path | None = None) -> SimConfig:
    """Load and validate the simulator configuration (default ``config/ust_sim.yaml``)."""
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    return SimConfig.model_validate(_flatten(_rename_keys(raw)))
