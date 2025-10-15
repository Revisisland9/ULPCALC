# src/model.py
import math
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd

ZONES_DEFAULT = ["Low", "Medium-Low", "Medium-High", "High"]

@dataclass
class CalibParams:
    target_mean: float = 0.18
    band_low: float = 0.10
    band_high: float = 0.40
    band_target: float = 0.95
    zones: int = 4
    iters: int = 80
    lr_mean: float = 0.25
    lr_tail: float = 0.12
    change_cap_pct: float = 0.05  # cap vs previous version (±%)

def _pick_column(cols, guess, fallback_idx):
    m = [c for c in cols if guess.lower() in str(c).lower()]
    return m[0] if m else cols[fallback_idx]

def load_dataframe(path: str, msrp_col: str | None = None, cost_col: str | None = None, state_col: str | None = None) -> pd.DataFrame:
    # Accept CSV or Excel
    if path.lower().endswith(".csv"):
        src = pd.read_csv(path)
    else:
        src = pd.read_excel(path)

    cols = list(src.columns)

    # Heuristics if explicit names aren't provided
    msrp_use  = msrp_col  or _pick_column(cols, "msrp",  0 if len(cols) > 0 else 0)
    cost_use  = cost_col  or _pick_column(cols, "cost to ulp", 1 if len(cols) > 1 else 0)
    state_use = state_col or _pick_column(cols, "destination state", 2 if len(cols) > 2 else 0)

    df = src.rename(columns={
        msrp_use:  "MSRP",
        cost_use:  "Cost_to_ULP",
        state_use: "Destination_State"
    }).copy()

    # Clean
    df["MSRP"] = pd.to_numeric(df["MSRP"], errors="coerce")
    df["Cost_to_ULP"] = pd.to_numeric(df["Cost_to_ULP"], errors="coerce")
    df["Destination_State"] = df["Destination_State"].astype(str).str.strip().str.upper()

    # Basic filters
    df = df.dropna(subset=["MSRP", "Cost_to_ULP", "Destination_State"])
    df = df[(df["MSRP"] > 0) & (df["Cost_to_ULP"] > 0)].copy()
    return df

def build_zones(df: pd.DataFrame, zones: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build state-level medians, assign zones by quantiles, and compute
    the per-zone expected cost ratio (count-weighted mean of state medians).
    These are the EXACT ratios used everywhere else (calibration + export).
    """
    # State medians of cost/MSRP
    state_stats = (
        df.assign(cost_ratio=df["Cost_to_ULP"] / df["MSRP"])
          .groupby("Destination_State")
          .agg(Count=("Destination_State", "size"),
               Median_Cost_Ratio=("cost_ratio", "median"))
          .reset_index()
    )

    # Zone edges by quantiles of state medians
    qcuts = np.linspace(0, 100, zones + 1)[1:-1]
    edges = np.percentile(state_stats["Median_Cost_Ratio"], qcuts) if len(state_stats) else []
    labels = ZONES_DEFAULT[:zones]

    def to_zone(v):
        idx = np.searchsorted(edges, v, side="right")
        return labels[idx]

    state_stats["Zone"] = state_stats["Median_Cost_Ratio"].apply(to_zone)

    # Per-zone expected cost ratio (count-weighted mean of state medians)
    zone_ratio = (
        state_stats.groupby("Zone")
        .apply(lambda g: np.average(g["Median_Cost_Ratio"], weights=g["Count"]))
        .rename("Zone_Expected_Cost_Ratio")
        .reset_index()
    )

    # Attach zone & expected ratio to each row
    dfz = df.merge(state_stats[["Destination_State", "Zone"]], on="Destination_State", how="left")
    dfz = dfz.merge(zone_ratio, on="Zone", how="left")

    return state_stats, zone_ratio, dfz

def _evaluate(dfz: pd.DataFrame, mults: Dict[str, float]) -> Tuple[pd.Series, pd.Series]:
    """
    Evaluate prices and margins using the SAME ratio that will be exported
    and used at quote time: Zone_Expected_Cost_Ratio (no hidden adjustments).
    """
    ratio_col = "Zone_Expected_Cost_Ratio"
    price = dfz["MSRP"] * dfz[ratio_col] * dfz["Zone"].map(mults)
    margin = (price - dfz["Cost_to_ULP"]) / dfz["Cost_to_ULP"]
    return price, margin

def calibrate(dfz: pd.DataFrame, params: CalibParams, prev_multipliers: Dict[str, float] | None = None) -> Dict[str, float]:
    """
    Tune per-zone multipliers so that:
      - overall mean margin approaches target_mean,
      - at least band_target of shipments fall in [band_low, band_high].
    Everything is computed against the exported ratio, so no drift.
    """
    zones = dfz["Zone"].dropna().unique().tolist()
    mults = {z: 1.0 + params.target_mean for z in zones}  # initialize near target

    for _ in range(params.iters):
        price, marg = _evaluate(dfz, mults)
        mean = float(marg.mean())

        # 1) Global centering toward the target mean
        if not math.isclose(mean, params.target_mean, rel_tol=1e-6, abs_tol=1e-6):
            scale = 1.0 + params.lr_mean * (params.target_mean - mean)
            for z in mults:
                mults[z] *= scale

        # 2) Per-zone tightening for band coverage
        zstats = (
            pd.DataFrame({"Zone": dfz["Zone"], "marg": marg})
              .groupby("Zone")["marg"]
              .agg(mean="mean",
                   below=lambda x: (x < params.band_low).mean(),
                   above=lambda x: (x > params.band_high).mean(),
                   inside=lambda x: ((x >= params.band_low) & (x <= params.band_high)).mean())
              .reset_index()
        )
        for _, r in zstats.iterrows():
            z = r["Zone"]
            if r["inside"] < params.band_target:
                if r["below"] > 0.05:
                    mults[z] *= (1 + params.lr_tail * min(0.10, r["below"]))
                if r["above"] > 0.05:
                    mults[z] *= (1 - params.lr_tail * min(0.10, r["above"]))

    # Optional cap vs previous version
    if prev_multipliers:
        capped = {}
        for z, m in mults.items():
            prev = prev_multipliers.get(z, m)
            hi = prev * (1 + params.change_cap_pct)
            lo = prev * (1 - params.change_cap_pct)
            capped[z] = max(lo, min(hi, m))
        mults = capped

    return mults

def normalize_total(dfz: pd.DataFrame, mults: Dict[str, float], params: CalibParams) -> Dict[str, float]:
    """
    Final step: ensure Σ(price) == Σ(cost) * (1 + target_mean),
    again using the SAME exported ratio.
    """
    price, _ = _evaluate(dfz, mults)
    target_rev = dfz["Cost_to_ULP"].sum() * (1 + params.target_mean)
    scale = float(target_rev / price.sum()) if price.sum() else 1.0
    return {z: m * scale for z, m in mults.items()}

def score(dfz: pd.DataFrame, mults: Dict[str, float], params: CalibParams) -> dict:
    price, marg = _evaluate(dfz, mults)
    inside = ((marg >= params.band_low) & (marg <= params.band_high)).mean()
    return {
        "total_cost": float(dfz["Cost_to_ULP"].sum()),
        "total_revenue": float(price.sum()),
        "mean_margin": float(marg.mean()),
        "pct_inside": float(inside),
        "pct_below": float((marg < params.band_low).mean()),
        "pct_above": float((marg > params.band_high).mean())
    }

def build_version(df: pd.DataFrame, params: CalibParams, prev: dict | None = None) -> dict:
    """
    Full pipeline: build zones/ratios, calibrate multipliers, normalize totals,
    score, and package a version payload for export. All steps use the same ratio
    (Zone_Expected_Cost_Ratio), which is exactly what the Quote page consumes.
    """
    state_stats, zone_ratio, dfz = build_zones(df, params.zones)

    prev_mults = None
    if prev:
        prev_mults = {r["zone"]: r["multiplier"] for r in prev.get("zones", [])}

    mults = calibrate(dfz, params, prev_mults)
    mults = normalize_total(dfz, mults, params)
    metrics = score(dfz, mults, params)

    # Build export tables aligned with the exact math we just used
    zone_table = zone_ratio.copy()
    zone_table["Zone_Multiplier"] = zone_table["Zone"].map(mults)

    state_map = state_stats[["Destination_State", "Zone"]].merge(
        zone_table[["Zone", "Zone_Expected_Cost_Ratio", "Zone_Multiplier"]],
        on="Zone", how="left"
    )

    version = {
        "params": vars(params),
        "metrics": metrics,
        "zones": [
            {
                "zone": row["Zone"],
                "expected_cost_ratio": float(row["Zone_Expected_Cost_Ratio"]),
                "multiplier": float(row["Zone_Multiplier"])
            }
            for _, row in zone_table.iterrows()
        ],
        "state_map": [
            {"state": r["Destination_State"], "zone": r["Zone"]}
            for _, r in state_stats.iterrows()
        ]
    }
    return version, state_map, zone_table
