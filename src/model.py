# src/model.py
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

ZONES_DEFAULT = ["Low", "Medium-Low", "Medium-High", "High"]

# ----------------------------
# Parameters
# ----------------------------
@dataclass
class CalibParams:
    # Global profitability target (mean margin on cost basis)
    target_mean: float = 0.18

    # Margin band we want most shipments to fall into
    band_low: float = 0.10
    band_high: float = 0.40
    band_target: float = 0.80  # per (zone×tier) cost-weighted coverage target

    # Zoning
    zones: int = 4  # number of state cost-behavior zones (quartiles by default)

    # MSRP tiering (piecewise). Keep your existing bins; you can change later.
    msrp_breaks: Tuple[float, ...] = (0.0, 500.0, 1000.0, 2000.0, float("inf"))

    # Stabilization (shrink zone ratio toward global ratio)
    shrinkage: float = 0.80  # 0.0 = global only, 1.0 = zone only

    # Optimizer
    iters: int = 120
    lr_mean: float = 0.30   # global centering step
    lr_tail: float = 0.15   # per (zone×tier) tightening step
    change_cap_pct: float = 0.07  # cap ± vs previous multipliers on publish

# ----------------------------
# Loader
# ----------------------------
def _pick_column(cols, guess, fallback_idx):
    m = [c for c in cols if guess.lower() in str(c).lower()]
    return m[0] if m else cols[fallback_idx]

def load_dataframe(path: str,
                   msrp_col: Optional[str] = None,
                   cost_col: Optional[str] = None,
                   state_col: Optional[str] = None) -> pd.DataFrame:
    """Load CSV/XLSX and map columns to MSRP, Cost_to_ULP, Destination_State explicitly."""
    if path.lower().endswith(".csv"):
        src = pd.read_csv(path)
    else:
        src = pd.read_excel(path)

    cols = list(src.columns)

    # Validate explicit selections if provided
    missing = []
    if msrp_col and msrp_col not in cols:   missing.append(f"MSRP='{msrp_col}'")
    if cost_col and cost_col not in cols:   missing.append(f"Cost to ULP='{cost_col}'")
    if state_col and state_col not in cols: missing.append(f"Destination State='{state_col}'")
    if missing:
        raise KeyError(f"Selected column(s) not found in file: {', '.join(missing)}. Available: {cols}")

    # Fallback heuristics if not provided
    msrp_use  = msrp_col  or _pick_column(cols, "msrp", 0 if len(cols)>0 else 0)
    cost_use  = cost_col  or _pick_column(cols, "cost to ulp", 1 if len(cols)>1 else 0)
    state_use = state_col or _pick_column(cols, "destination state", 2 if len(cols)>2 else 0)

    df = src.rename(columns={
        msrp_use: "MSRP",
        cost_use: "Cost_to_ULP",
        state_use: "Destination_State"
    }).copy()

    # Clean
    df["MSRP"] = pd.to_numeric(df["MSRP"], errors="coerce")
    df["Cost_to_ULP"] = pd.to_numeric(df["Cost_to_ULP"], errors="coerce")
    df["Destination_State"] = df["Destination_State"].astype(str).str.strip().str.upper()

    # Filters
    df = df.dropna(subset=["MSRP", "Cost_to_ULP", "Destination_State"])
    df = df[(df["MSRP"] > 0) & (df["Cost_to_ULP"] > 0)].copy()
    return df

# ----------------------------
# Zoning & Tiering
# ----------------------------
def _assign_zones(df: pd.DataFrame, zones: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build state medians of cost/MSRP, assign states to zones by quantiles,
       and compute zone expected cost ratios (count-weighted across states)."""
    state_stats = (
        df.assign(cost_ratio=df["Cost_to_ULP"]/df["MSRP"])
          .groupby("Destination_State")
          .agg(Count=("Destination_State","size"),
               Median_Cost_Ratio=("cost_ratio","median"))
          .reset_index()
    )

    qcuts = np.linspace(0, 100, zones + 1)[1:-1]
    edges = np.percentile(state_stats["Median_Cost_Ratio"], qcuts) if len(state_stats) else []
    labels = ZONES_DEFAULT[:zones]

    def to_zone(v):
        idx = np.searchsorted(edges, v, side="right")
        return labels[idx]

    state_stats["Zone"] = state_stats["Median_Cost_Ratio"].apply(to_zone)

    # Count-weighted mean of state medians per zone
    zone_ratio = (
        state_stats.groupby("Zone")
        .apply(lambda g: np.average(g["Median_Cost_Ratio"], weights=g["Count"]))
        .rename("Zone_Expected_Cost_Ratio")
        .reset_index()
    )
    return state_stats, zone_ratio

def _assign_tiers(msrp: pd.Series, breaks: Tuple[float, ...]) -> Tuple[pd.Series, List[str], List[float]]:
    bins = list(breaks)
    labels = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        labels.append(f"{int(lo)}–{('∞' if hi==float('inf') else int(hi))}")
    idx = pd.cut(msrp, bins=bins, right=False, labels=labels, include_lowest=True)
    return idx.astype(str), labels, bins

# ----------------------------
# Weighted quantiles
# ----------------------------
def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumw = np.cumsum(w)
    if cumw[-1] <= 0:
        return float("nan")
    return float(np.interp(q * cumw[-1], cumw, v))

# ----------------------------
# Core evaluation helpers
# ----------------------------
def _evaluate(dfz: pd.DataFrame,
              mults_ab: Dict[Tuple[str, str], Tuple[float, float]],
              ratio_col: str) -> Tuple[pd.Series, pd.Series]:
    """Compute price and realized margin using a two-parameter multiplier per (Zone × Tier).
       m = a + b * MSRP_Norm
    """
    key = list(zip(dfz["Zone"], dfz["Tier"]))
    a = np.array([mults_ab.get(k, (1.0, 0.0))[0] for k in key], dtype=float)
    b = np.array([mults_ab.get(k, (1.0, 0.0))[1] for k in key], dtype=float)
    m = a + b * dfz["MSRP_Norm"].values
    # small safety clamp on m
    m = np.clip(m, 0.2, 5.0)
    price = dfz["MSRP"].values * dfz[ratio_col].values * m
    margin = (price - dfz["Cost_to_ULP"].values) / dfz["Cost_to_ULP"].values
    return pd.Series(price, index=dfz.index), pd.Series(margin, index=dfz.index)

# ----------------------------
# Calibration prep
# ----------------------------
def build_zones_and_tiers(df: pd.DataFrame, params: CalibParams) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    """Attach zone and tier to each row; compute zone expected ratios (with shrinkage)."""
    state_stats, zone_ratio = _assign_zones(df, params.zones)

    # Global ratio for shrinkage baseline
    global_ratio = df["Cost_to_ULP"].sum() / df["MSRP"].sum()

    # Attach zone to rows
    dfz = df.merge(state_stats[["Destination_State", "Zone"]], on="Destination_State", how="left")
    dfz = dfz.merge(zone_ratio, on="Zone", how="left")

    # Apply shrinkage toward global ratio (stabilizes extremes)
    dfz["Adj_Zone_Ratio"] = params.shrinkage * dfz["Zone_Expected_Cost_Ratio"] + (1 - params.shrinkage) * global_ratio

    # Assign MSRP tier labels
    tier_series, tier_labels, breaks = _assign_tiers(dfz["MSRP"], params.msrp_breaks)
    dfz["Tier"] = tier_series

    # Compute per-tier normalization stats (weighted by cost):
    # mid = weighted median MSRP; width = max(1, p90 - p10) to robustly span each bin
    tnorm_rows = []
    for lab in tier_labels:
        mask = (dfz["Tier"] == lab)
        g = dfz.loc[mask]
        if g.empty:
            mid = 0.5 * (breaks[tier_labels.index(lab)] + breaks[tier_labels.index(lab)+1 if np.isfinite(breaks[tier_labels.index(lab)+1]) else tier_labels.index(lab)] if len(breaks)>1 else 0.0)
            width = 1.0
        else:
            w = g["Cost_to_ULP"].values
            v = g["MSRP"].values
            mid = _weighted_quantile(v, w, 0.5)
            p10 = _weighted_quantile(v, w, 0.10)
            p90 = _weighted_quantile(v, w, 0.90)
            width = max(1.0, p90 - p10)
        tnorm_rows.append({"tier": lab, "mid": float(mid), "width": float(width)})

    tnorm = pd.DataFrame(tnorm_rows)

    # Normalize MSRP within tier: x = (MSRP - mid)/width
    mid_map = dict(zip(tnorm["tier"], tnorm["mid"]))
    wid_map = dict(zip(tnorm["tier"], tnorm["width"]))
    dfz["Tier_Mid"] = dfz["Tier"].map(mid_map).astype(float)
    dfz["Tier_Width"] = dfz["Tier"].map(wid_map).astype(float)
    dfz["MSRP_Norm"] = (dfz["MSRP"] - dfz["Tier_Mid"]) / dfz["Tier_Width"]

    return state_stats, zone_ratio, dfz, global_ratio, tnorm

# ----------------------------
# Calibration (learn a,b

