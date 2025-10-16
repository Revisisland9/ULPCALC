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
    band_target: float = 0.95  # cost-weighted coverage target

    # Zoning
    zones: int = 4  # number of state cost-behavior zones (quartiles by default)

    # MSRP tiering (piecewise)
    # Breaks define bins: [0, 500, 1000, 2000, +inf] → 4 tiers by default
    msrp_breaks: Tuple[float, ...] = (0.0, 500.0, 1000.0, 2000.0, float("inf"))

    # Stabilization (shrink zone ratio toward global ratio)
    shrinkage: float = 0.80  # 0.0 = global only, 1.0 = zone only

    # Optimizer
    iters: int = 90
    lr_mean: float = 0.25   # global centering step
    lr_tail: float = 0.10   # per (zone×tier) tightening step
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
    """Load CSV/XLSX and map columns to MSRP, Cost_to_ULP, Destination_State explicitly.
       Raises a clear error if a selected column is missing.
    """
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

def _assign_tiers(msrp: pd.Series, breaks: Tuple[float, ...]) -> pd.Series:
    bins = list(breaks)
    labels = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        labels.append(f"{int(lo)}–{('∞' if hi==float('inf') else int(hi))}")
    idx = pd.cut(msrp, bins=bins, right=False, labels=labels, include_lowest=True)
    return idx.astype(str)

# ----------------------------
# Core evaluation helpers
# ----------------------------
def _evaluate(dfz: pd.DataFrame,
              mults: Dict[Tuple[str, str], float],
              ratio_col: str) -> Tuple[pd.Series, pd.Series]:
    """Compute price and realized margin using a tiered multiplier per (Zone × Tier)."""
    key = list(zip(dfz["Zone"], dfz["Tier"]))
    m = np.array([mults.get(k, 1.0) for k in key], dtype=float)
    price = dfz["MSRP"] * dfz[ratio_col].values * m
    margin = (price - dfz["Cost_to_ULP"].values) / dfz["Cost_to_ULP"].values
    return pd.Series(price, index=dfz.index), pd.Series(margin, index=dfz.index)

# ----------------------------
# Calibration
# ----------------------------
def build_zones_and_tiers(df: pd.DataFrame, params: CalibParams) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
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
    dfz["Tier"] = _assign_tiers(dfz["MSRP"], params.msrp_breaks)

    return state_stats, zone_ratio, dfz, global_ratio

def calibrate(dfz: pd.DataFrame,
              params: CalibParams,
              prev_multipliers: Optional[Dict[Tuple[str,str], float]] = None) -> Dict[Tuple[str,str], float]:
    """Learn multipliers per (Zone × Tier) to hit the global mean and tighten band coverage."""
    # Start near target for all zone×tier keys observed
    keys = sorted(set(zip(dfz["Zone"], dfz["Tier"])))
    mults = {k: 1.0 + params.target_mean for k in keys}

    # cost-weight vector
    w = dfz["Cost_to_ULP"].values
    w = np.where(w > 0, w, 1.0)

    ratio_col = "Adj_Zone_Ratio"  # use stabilized ratio for optimization

    for _ in range(params.iters):
        price, marg = _evaluate(dfz, mults, ratio_col=ratio_col)

        # 1) Global centering towards target revenue (Σprice = Σcost × (1+target))
        target_rev = dfz["Cost_to_ULP"].sum() * (1 + params.target_mean)
        cur_rev = float(price.sum())
        if cur_rev != 0:
            scale = target_rev / cur_rev
            step = 1.0 + params.lr_mean * (scale - 1.0)
            for k in mults:
                mults[k] *= step

        # 2) Per (Zone×Tier) tightening for cost-weighted band coverage
        ztdf = pd.DataFrame({
            "Zone": dfz["Zone"].values,
            "Tier": dfz["Tier"].values,
            "marg": marg.values,
            "w": w
        })
        grp = ztdf.groupby(["Zone","Tier"])
        tot_w = grp["w"].sum()
        below_w = grp.apply(lambda g: g.loc[g["marg"] < params.band_low, "w"].sum())
        above_w = grp.apply(lambda g: g.loc[g["marg"] > params.band_high, "w"].sum())
        inside_w = tot_w - below_w - above_w
        # ratios
        below_r = (below_w / tot_w).fillna(0.0)
        above_r = (above_w / tot_w).fillna(0.0)
        inside_r = (inside_w / tot_w).fillna(0.0)

        for (z,t), _tot in tot_w.items():
            if _tot <= 0: 
                continue
            key = (z,t)
            if inside_r.loc[(z,t)] < params.band_target:
                if below_r.loc[(z,t)] > 0.04:
                    mults[key] *= (1 + params.lr_tail * min(0.10, float(below_r.loc[(z,t)])))
                if above_r.loc[(z,t)] > 0.04:
                    mults[key] *= (1 - params.lr_tail * min(0.10, float(above_r.loc[(z,t)])))

    # Optional cap vs previous version (stability)
    if prev_multipliers:
        capped = {}
        for k, m in mults.items():
            prev = prev_multipliers.get(k, m)
            hi = prev * (1 + params.change_cap_pct)
            lo = prev * (1 - params.change_cap_pct)
            capped[k] = max(lo, min(hi, m))
        mults = capped

    return mults

def normalize_total(dfz: pd.DataFrame,
                    mults: Dict[Tuple[str,str], float],
                    params: CalibParams) -> Dict[Tuple[str,str], float]:
    """Final normalization: ensure Σprice == Σcost × (1+target), with stabilized ratios."""
    price, _ = _evaluate(dfz, mults, ratio_col="Adj_Zone_Ratio")
    target_rev = dfz["Cost_to_ULP"].sum() * (1 + params.target_mean)
    scale = float(target_rev / price.sum()) if price.sum() else 1.0
    return {k: m * scale for k, m in mults.items()}

# ----------------------------
# Scoring
# ----------------------------
def score(dfz: pd.DataFrame,
          mults: Dict[Tuple[str,str], float],
          params: CalibParams) -> dict:
    """Return cost-weighted KPIs (primary) and unweighted for reference."""
    price, marg = _evaluate(dfz, mults, ratio_col="Adj_Zone_Ratio")

    # primary (cost-weighted)
    total_cost = float(dfz["Cost_to_ULP"].sum())
    total_rev  = float(price.sum())
    mean_w = (total_rev / total_cost - 1.0) if total_cost > 0 else float("nan")
    w = dfz["Cost_to_ULP"].values
    w = np.where(w > 0, w, 1.0)
    wsum = w.sum() if w.sum() != 0 else 1.0

    inside_w = float(w[((marg>=params.band_low)&(marg<=params.band_high))].sum() / wsum)
    below_w  = float(w[(marg < params.band_low)].sum() / wsum)
    above_w  = float(w[(marg > params.band_high)].sum() / wsum)

    # reference (unweighted)
    mean_unw = float(marg.mean())
    inside_unw = float(((marg>=params.band_low)&(marg<=params.band_high)).mean())
    below_unw  = float((marg<params.band_low).mean())
    above_unw  = float((marg>params.band_high).mean())

    return {
        "total_cost": total_cost,
        "total_revenue": total_rev,
        "mean_margin": mean_w,
        "pct_inside": inside_w,
        "pct_below": below_w,
        "pct_above": above_w,
        "mean_margin_unweighted": mean_unw,
        "pct_inside_unweighted": inside_unw,
        "pct_below_unweighted": below_unw,
        "pct_above_unweighted": above_unw
    }

# ----------------------------
# Build version payload (export)
# ----------------------------
def build_version(df: pd.DataFrame,
                  params: CalibParams,
                  prev: Optional[dict] = None) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Full pipeline: zone & tier assignment, calibration, normalization, scoring, export tables."""
    state_stats, zone_ratio, dfz, global_ratio = build_zones_and_tiers(df, params)

    # Previous multipliers (for change cap), if available
    prev_mults = None
    if prev and "zt_multipliers" in prev:
        prev_mults = { (r["zone"], r["tier"]): r["multiplier"] for r in prev["zt_multipliers"] }

    mults = calibrate(dfz, params, prev_multipliers=prev_mults)
    mults = normalize_total(dfz, mults, params)
    metrics = score(dfz, mults, params)

    # Build exportables

    # 1) Zone table (with stabilized ratio used for pricing/export)
    # We export the STABILIZED ratio per zone so the Quote/Bulk pages match calibration.
    # Compute stabilized per-zone ratio by averaging row-level Adj_Zone_Ratio within each zone (cost-weighted).
    zdf = dfz.groupby("Zone").apply(
        lambda g: pd.Series({
            "Zone_Expected_Cost_Ratio": float(np.average(g["Adj_Zone_Ratio"], weights=g["Cost_to_ULP"]))
        })
    ).reset_index()
    zone_table = zdf.copy()
    zone_table["Zone"] = zone_table["Zone"].astype(str)

    # 2) State map (state → zone)
    state_map = state_stats[["Destination_State","Zone"]].copy()

    # 3) Tier definitions (edges/labels)
    breaks = list(params.msrp_breaks)
    tier_labels = []
    for i in range(len(breaks)-1):
        lo, hi = breaks[i], breaks[i+1]
        tier_labels.append(f"{int(lo)}–{('∞' if hi==float('inf') else int(hi))}")

    # 4) Zone×Tier multipliers
    zt_rows = []
    for (z, t), m in sorted(mults.items()):
        zt_rows.append({
            "zone": str(z),
            "tier": str(t),
            "multiplier": float(m)
        })

    # Final version payload
    version = {
        "params": {
            **vars(params),
            "msrp_breaks": list(params.msrp_breaks)
        },
        "metrics": metrics,
        # Stabilized ratios per zone (used everywhere)
        "zones": [
            {"zone": r["Zone"], "expected_cost_ratio": float(r["Zone_Expected_Cost_Ratio"])}
            for _, r in zone_table.iterrows()
        ],
        # State → zone lookup
        "state_map": [
            {"state": r["Destination_State"], "zone": r["Zone"]}
            for _, r in state_map.iterrows()
        ],
        # MSRP tier edges and labels
        "tiers": {
            "breaks": breaks,
            "labels": tier_labels
        },
        # Core: multipliers per (zone × tier)
        "zt_multipliers": zt_rows
    }

    return version, state_map, zone_table

# ----------------------------
# Runtime helpers (for Quote/Bulk)
# ----------------------------
def _tier_for_value(v: float, breaks: List[float], labels: List[str]) -> str:
    for i in range(len(breaks)-1):
        if breaks[i] <= v < breaks[i+1]:
            return labels[i]
    return labels[-1]

def _mult_lookup(version: dict, zone: str, tier: str) -> float:
    # Build an index on first call and cache it in the version dict
    if "_mult_index" not in version:
        idx = {}
        for row in version.get("zt_multipliers", []):
            idx[(row["zone"], row["tier"])] = row["multiplier"]
        version["_mult_index"] = idx
    return version["_mult_index"].get((zone, tier), 1.0 + version["params"].get("target_mean", 0.18))

def price_for(msrp: float, state: str, version: dict) -> float:
    """Compute price for a single (MSRP, state) using the tiered model."""
    state = str(state).strip().upper()
    # map state -> zone
    smap = version.get("_state_index")
    if smap is None:
        smap = { r["state"]: r["zone"] for r in version.get("state_map", []) }
        version["_state_index"] = smap
    zone = smap.get(state)
    if zone is None:
        # fallback: use most common (or first) zone if unseen
        zone = version["zones"][0]["zone"]

    # zone stabilized expected ratio
    zrat = { r["zone"]: r["expected_cost_ratio"] for r in version.get("zones", []) }.get(zone, 1.0)

    # tier
    breaks = version["tiers"]["breaks"]
    labels = version["tiers"]["labels"]
    tier = _tier_for_value(float(msrp), breaks, labels)

    mult = _mult_lookup(version, zone, tier)
    return float(msrp) * float(zrat) * float(mult)

def price_for_many(msrps: List[float], states: List[str], version: dict) -> List[float]:
    return [price_for(msrp, state, version) for msrp, state in zip(msrps, states)]
