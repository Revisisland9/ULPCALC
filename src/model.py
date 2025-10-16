# src/model.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

ZONES_DEFAULT = ["Low", "Medium-Low", "Medium-High", "High"]

# ============================
# Parameters
# ============================
@dataclass
class CalibParams:
    # Profitability & band
    target_mean: float = 0.18
    band_low: float = 0.10
    band_high: float = 0.40
    band_target: float = 0.85  # optimization tries to reach this coverage
    # Zoning / tiering
    zones: int = 4
    msrp_breaks: Optional[Tuple[float, ...]] = None  # None → use quantile tiers
    tiers: int = 16                                  # number of quantile tiers if msrp_breaks is None
    shrinkage: float = 0.70                          # zone ratio shrink toward global
    # Optimizer
    iters: int = 140
    lr_mean: float = 0.30
    lr_tail: float = 0.22
    change_cap_pct: float = 0.07
    # Slope clamp for b
    b_cap: float = 1.0
    # Per-state correction
    with_c_state: bool = True
    c_state_step_scale: float = 0.03   # tiny step
    c_state_decay: float = 0.90        # shrink toward 0 each iter
    c_state_cap: float = 0.12          # clamp magnitude of c_state
    # Optional runtime guard (applied in price_for)
    guard_low: Optional[float] = None  # e.g., 0.10
    guard_high: Optional[float] = None # e.g., 0.40

# ============================
# Loader (paths & file-like)
# ============================
def _pick_column(cols: List[str], guess, fallback_idx: int) -> str:
    try:
        g = str(guess).lower()
    except Exception:
        g = ""
    matches = [c for c in cols if g in str(c).lower()]
    if matches:
        return matches[0]
    if not cols:
        raise KeyError("No columns found in source file.")
    return cols[min(max(fallback_idx, 0), len(cols) - 1)]

def load_dataframe(source,
                   msrp_col: Optional[str] = None,
                   cost_col: Optional[str] = None,
                   state_col: Optional[str] = None) -> pd.DataFrame:
    """Load CSV/XLSX from a path or a file-like (Streamlit UploadedFile)."""
    if isinstance(source, str):
        name = source.lower()
        if name.endswith(".csv"):
            src = pd.read_csv(source)
        else:
            src = pd.read_excel(source)
    else:
        name = str(getattr(source, "name", "")).lower()
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            if name.endswith(".csv"):
                src = pd.read_csv(source)
            else:
                src = pd.read_excel(source)
        finally:
            if hasattr(source, "seek"):
                source.seek(0)

    cols = list(src.columns)
    missing = []
    if msrp_col and msrp_col not in cols:   missing.append(f"MSRP='{msrp_col}'")
    if cost_col and cost_col not in cols:   missing.append(f"Cost to ULP='{cost_col}'")
    if state_col and state_col not in cols: missing.append(f"Destination State='{state_col}'")
    if missing:
        raise KeyError(f"Missing columns: {', '.join(missing)}. Found: {cols}")

    msrp_use  = msrp_col  or _pick_column(cols, "msrp", 0)
    cost_use  = cost_col  or _pick_column(cols, "cost to ulp", 1 if len(cols) > 1 else 0)
    state_use = state_col or _pick_column(cols, "destination state", 2 if len(cols) > 2 else 0)

    df = src.rename(columns={
        msrp_use: "MSRP",
        cost_use: "Cost_to_ULP",
        state_use: "Destination_State"
    }).copy()

    df["MSRP"] = pd.to_numeric(df["MSRP"], errors="coerce")
    df["Cost_to_ULP"] = pd.to_numeric(df["Cost_to_ULP"], errors="coerce")
    df["Destination_State"] = df["Destination_State"].astype(str).str.strip().str.upper()

    df = df.dropna(subset=["MSRP", "Cost_to_ULP", "Destination_State"])
    df = df[(df["MSRP"] > 0) & (df["Cost_to_ULP"] > 0)].copy()
    return df

# ============================
# Zoning & tiering
# ============================
def _zone_labels(n: int) -> List[str]:
    base = ZONES_DEFAULT[:]
    if n <= len(base):
        return base[:n]
    return base + [f"Z{i}" for i in range(5, n + 1)]

def _assign_zones(df: pd.DataFrame, zones: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    state_stats = (
        df.assign(cost_ratio=df["Cost_to_ULP"]/df["MSRP"])
          .groupby("Destination_State")
          .agg(Count=("Destination_State","size"),
               Median_Cost_Ratio=("cost_ratio","median"))
          .reset_index()
    )
    qcuts = np.linspace(0, 100, zones + 1)[1:-1]
    edges = np.percentile(state_stats["Median_Cost_Ratio"], qcuts) if len(state_stats) else []
    labels = _zone_labels(zones)
    def to_zone(v):
        idx = np.searchsorted(edges, v, side="right")
        if idx >= zones: idx = zones - 1
        return labels[idx]
    state_stats["Zone"] = state_stats["Median_Cost_Ratio"].apply(to_zone)

    zone_ratio = (
        state_stats.groupby("Zone")
        .apply(lambda g: np.average(g["Median_Cost_Ratio"], weights=g["Count"]))
        .rename("Zone_Expected_Cost_Ratio")
        .reset_index()
    )
    return state_stats, zone_ratio

def _quantile_breaks(series: pd.Series, n: int) -> Tuple[List[float], List[str]]:
    qs = np.linspace(0, 1, n+1)
    edges = np.unique(series.quantile(qs).values.astype(float))
    if len(edges) < 2:
        edges = np.array([0.0, float("inf")])
    edges[0] = 0.0
    edges[-1] = float("inf")
    labels = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        labels.append(f"{int(lo)}–{('∞' if hi==float('inf') else int(hi))}")
    return list(edges), labels

def _assign_tiers(msrp: pd.Series,
                  msrp_breaks: Optional[Tuple[float, ...]],
                  tiers: int) -> Tuple[pd.Series, List[str], List[float]]:
    if msrp_breaks is None:
        breaks, labels = _quantile_breaks(msrp, tiers)
    else:
        breaks = list(msrp_breaks)
        labels = []
        for i in range(len(breaks)-1):
            lo, hi = breaks[i], breaks[i+1]
            labels.append(f"{int(lo)}–{('∞' if hi==float('inf') else int(hi))}")
    idx = pd.cut(msrp, bins=breaks, right=False, labels=labels, include_lowest=True)
    return idx.astype(str), labels, breaks

def _weighted_quantile(values, weights, q):
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumw = np.cumsum(w)
    if cumw[-1] <= 0:
        return float("nan")
    return float(np.interp(q * cumw[-1], cumw, v))

# ============================
# Core evaluation
# ============================
def _evaluate(dfz: pd.DataFrame,
              mults_ab: Dict[Tuple[str,str], Tuple[float,float]],
              c_state: Optional[Dict[str,float]] = None,
              b_cap: float = 1.0,
              guard: Optional[Tuple[float,float]] = None) -> Tuple[pd.Series, pd.Series]:
    key = list(zip(dfz["Zone"], dfz["Tier"]))
    a = np.array([mults_ab.get(k, (1.0, 0.0))[0] for k in key], float)
    b = np.array([mults_ab.get(k, (1.0, 0.0))[1] for k in key], float)
    b = np.clip(b, -b_cap, b_cap)
    m = np.clip(a + b * dfz["MSRP_Norm"].values, 0.2, 5.0)
    if c_state is not None:
        c = dfz["Destination_State"].map(c_state).fillna(0.0).values
        m = np.clip(m + c, 0.2, 5.0)

    price = dfz["MSRP"].values * dfz["Adj_Zone_Ratio"].values * m

    # Optional band guard at price time
    if guard is not None:
        low, high = guard
        cost = dfz["Cost_to_ULP"].values
        price = np.minimum(np.maximum(price, cost*(1.0 + low)), cost*(1.0 + high))

    margin = (price - dfz["Cost_to_ULP"].values) / dfz["Cost_to_ULP"].values
    return pd.Series(price, index=dfz.index), pd.Series(margin, index=dfz.index)

# ============================
# Build zones/tiers + normalization features
# ============================
def build_zones_and_tiers(df: pd.DataFrame, params: CalibParams):
    state_stats, zone_ratio = _assign_zones(df, params.zones)
    global_ratio = df["Cost_to_ULP"].sum() / df["MSRP"].sum()

    dfz = df.merge(state_stats[["Destination_State","Zone"]], on="Destination_State", how="left")
    dfz = dfz.merge(zone_ratio, on="Zone", how="left")

    # Stabilized zone cost ratio used for pricing
    dfz["Adj_Zone_Ratio"] = params.shrinkage * dfz["Zone_Expected_Cost_Ratio"] + (1 - params.shrinkage) * global_ratio

    # Tiers
    tier_series, tier_labels, breaks = _assign_tiers(dfz["MSRP"], params.msrp_breaks, params.tiers)
    dfz["Tier"] = tier_series

    # Tier normalization (mid, width)
    rows = []
    for lab in tier_labels:
        g = dfz[dfz["Tier"] == lab]
        if g.empty:
            mid, width = 0.0, 1.0
        else:
            w = g["Cost_to_ULP"].values
            v = g["MSRP"].values
            mid = _weighted_quantile(v, w, 0.5)
            p10 = _weighted_quantile(v, w, 0.10)
            p90 = _weighted_quantile(v, w, 0.90)
            width = max(1.0, p90 - p10)
        rows.append({"tier": lab, "mid": float(mid), "width": float(width)})

    tnorm = pd.DataFrame(rows)
    mid_map = dict(zip(tnorm["tier"], tnorm["mid"]))
    wid_map = dict(zip(tnorm["tier"], tnorm["width"]))
    dfz["Tier_Mid"] = dfz["Tier"].map(mid_map)
    dfz["Tier_Width"] = dfz["Tier"].map(wid_map)
    dfz["MSRP_Norm"] = (dfz["MSRP"] - dfz["Tier_Mid"]) / dfz["Tier_Width"]

    return state_stats, zone_ratio, dfz, tnorm, breaks, tier_labels, global_ratio

# ============================
# Calibration (a,b) + c_state
# ============================
def calibrate(dfz: pd.DataFrame,
              params: CalibParams,
              prev_multipliers: Optional[Dict[Tuple[str,str], Tuple[float,float]]] = None):
    keys = sorted(set(zip(dfz["Zone"], dfz["Tier"])))
    mults = {k: (1.0 + params.target_mean, 0.0) for k in keys}
    w = np.where(dfz["Cost_to_ULP"] > 0, dfz["Cost_to_ULP"], 1.0)

    # Initialize per-state correction (small & shrunk)
    c_state = {s: 0.0 for s in dfz["Destination_State"].unique()} if params.with_c_state else None

    for _ in range(params.iters):
        price, marg = _evaluate(dfz, mults, c_state=c_state, b_cap=params.b_cap)

        # Global centering to target mean
        target_rev = dfz["Cost_to_ULP"].sum() * (1 + params.target_mean)
        cur_rev = float(price.sum())
        if cur_rev != 0:
            step = 1.0 + params.lr_mean * ((target_rev / cur_rev) - 1.0)
            for k,(a,b) in mults.items():
                mults[k] = (a*step, b*step)

        # Per (Zone×Tier) tightening
        gdf = pd.DataFrame({
            "Zone": dfz["Zone"], "Tier": dfz["Tier"],
            "marg": marg, "w": w, "x": dfz["MSRP_Norm"]
        })
        for (z,t), g in gdf.groupby(["Zone","Tier"]):
            a,b = mults[(z,t)]
            totw = float(g["w"].sum())
            if totw <= 0:
                continue
            below_w = float(g.loc[g["marg"] < params.band_low,  "w"].sum())
            above_w = float(g.loc[g["marg"] > params.band_high, "w"].sum())
            inside_w = totw - below_w - above_w
            inside_r = inside_w / totw

            if inside_r < params.band_target:
                # shift center against heavier tail
                skew = (below_w - above_w) / totw
                a *= (1.0 + params.lr_tail * skew)

                # tilt: correlate outside-direction with MSRP_Norm
                outsig = np.zeros(len(g))
                outsig[g["marg"] < params.band_low]  = +1.0
                outsig[g["marg"] > params.band_high] = -1.0
                x   = g["x"].values
                wgt = g["w"].values
                wx  = np.average(x,    weights=wgt)
                wo  = np.average(outsig, weights=wgt)
                cov = float(np.average((x - wx) * (outsig - wo), weights=wgt))
                b = float(np.clip(b + params.lr_tail * 0.5 * cov, -params.b_cap, params.b_cap))

            mults[(z,t)] = (a,b)

        # Per-state correction (small EMA with heavy shrink)
        if c_state is not None:
            outsig_all = np.where(marg < params.band_low, +1.0, 0.0) + np.where(marg > params.band_high, -1.0, 0.0)
            sdf = pd.DataFrame({"state": dfz["Destination_State"], "w": w, "outsig": outsig_all})
            smean = sdf.groupby("state").apply(lambda g: float(np.average(g["outsig"], weights=g["w"]))).to_dict()
            for s, v in smean.items():
                prev = c_state.get(s, 0.0)
                c_state[s] = params.c_state_decay * prev + (params.lr_tail * params.c_state_step_scale) * float(v)
                c_state[s] = float(np.clip(c_state[s], -params.c_state_cap, params.c_state_cap))

    # Optional cap vs previous version (stability)
    if prev_multipliers:
        capped = {}
        for k, (a,b) in mults.items():
            prev_a, prev_b = prev_multipliers.get(k, (a,b))
            hi_a, lo_a = prev_a*(1+params.change_cap_pct), prev_a*(1-params.change_cap_pct)
            hi_b, lo_b = prev_b*(1+params.change_cap_pct), prev_b*(1-params.change_cap_pct)
            capped[k] = (max(lo_a, min(hi_a, a)), max(lo_b, min(hi_b, b)))
        mults = capped

    return mults, c_state

def normalize_total(dfz: pd.DataFrame,
                    mults: Dict[Tuple[str,str], Tuple[float,float]],
                    c_state: Optional[Dict[str,float]],
                    params: CalibParams) -> Dict[Tuple[str,str], Tuple[float,float]]:
    """Final normalization: hit exact target Σprice == Σcost × (1+target)."""
    price, _ = _evaluate(dfz, mults, c_state=c_state, b_cap=params.b_cap)
    target_rev = dfz["Cost_to_ULP"].sum() * (1 + params.target_mean)
    scale = float(target_rev / price.sum()) if price.sum() else 1.0
    return {k: (a*scale, b*scale) for k,(a,b) in mults.items()}

# ============================
# Scoring
# ============================
def score(dfz: pd.DataFrame,
          mults: Dict[Tuple[str,str], Tuple[float,float]],
          params: CalibParams,
          c_state: Optional[Dict[str,float]] = None,
          apply_guard: bool = False) -> dict:
    guard = (params.band_low, params.band_high) if apply_guard and params.guard_low is None else None
    if apply_guard and (params.guard_low is not None and params.guard_high is not None):
        guard = (params.guard_low, params.guard_high)

    price, marg = _evaluate(dfz, mults, c_state=c_state, b_cap=params.b_cap, guard=guard)

    total_cost = float(dfz["Cost_to_ULP"].sum())
    total_rev  = float(price.sum())
    mean_w = (total_rev / total_cost - 1.0) if total_cost > 0 else float("nan")

    w = np.where(dfz["Cost_to_ULP"] > 0, dfz["Cost_to_ULP"], 1.0)
    wsum = w.sum() or 1.0
    inside_w = float(w[(marg >= params.band_low) & (marg <= params.band_high)].sum() / wsum)
    below_w  = float(w[marg < params.band_low].sum() / wsum)
    above_w  = float(w[marg > params.band_high].sum() / wsum)
    loss_w   = float(w[marg < 0].sum() / wsum)

    return {
        "total_cost": total_cost,
        "total_revenue": total_rev,
        "mean_margin": mean_w,
        "pct_inside": inside_w,
        "pct_below": below_w,
        "pct_above": above_w,
        "pct_loss": loss_w
    }

# ============================
# Build version payload (export)
# ============================
def build_version(df: pd.DataFrame,
                  params: CalibParams,
                  prev: Optional[dict] = None) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    state_stats, zone_ratio, dfz, tnorm, breaks, labels, global_ratio = build_zones_and_tiers(df, params)

    prev_mults = None
    if prev and "zt_multipliers" in prev:
        prev_mults = {}
        for r in prev["zt_multipliers"]:
            if "a" in r and "b" in r:
                prev_mults[(r["zone"], r["tier"])] = (float(r["a"]), float(r["b"]))
            elif "multiplier" in r:
                prev_mults[(r["zone"], r["tier"])] = (float(r["multiplier"]), 0.0)

    mults, c_state = calibrate(dfz, params, prev_multipliers=prev_mults)
    mults = normalize_total(dfz, mults, c_state, params)

    # Metrics (without and with guard so you can preview both)
    metrics_no_guard = score(dfz, mults, params, c_state=c_state, apply_guard=False)
    metrics_guard    = score(dfz, mults, params, c_state=c_state, apply_guard=True)

    # Zone table (stabilized ratio actually used)
    zone_table = dfz.groupby("Zone").apply(
        lambda g: pd.Series({"Zone_Expected_Cost_Ratio":
                             float(np.average(g["Adj_Zone_Ratio"], weights=g["Cost_to_ULP"]))})
    ).reset_index()

    state_map = state_stats[["Destination_State","Zone"]].copy()

    # Serialize multipliers/state adjust
    zt_rows = [{"zone": z, "tier": t, "a": float(a), "b": float(b)} for (z,t),(a,b) in sorted(mults.items())]
    state_adjust = [{"state": s, "c": float(c)} for s,c in sorted((c_state or {}).items())]

    version = {
        "params": {
            **vars(params),
            "msrp_breaks": list(params.msrp_breaks) if params.msrp_breaks is not None else None
        },
        "metrics": {
            "no_guard": metrics_no_guard,
            "with_guard": metrics_guard
        },
        "zones": [
            {"zone": r["Zone"], "expected_cost_ratio": float(r["Zone_Expected_Cost_Ratio"])}
            for _, r in zone_table.iterrows()
        ],
        "state_map": [
            {"state": r["Destination_State"], "zone": r["Zone"]}
            for _, r in state_map.iterrows()
        ],
        "tiers": {
            "breaks": breaks,
            "labels": labels
        },
        "tier_norm": [
            {"tier": r["tier"], "mid": float(r["mid"]), "width": float(r["width"])}
            for _, r in tnorm.iterrows()
        ],
        "zt_multipliers": zt_rows,
        "state_adjust": state_adjust,
        # If guard_low/high are set in params, persist a guard block for runtime
        "guard": (
            {"low": params.guard_low, "high": params.guard_high}
            if (params.guard_low is not None and params.guard_high is not None)
            else None
        )
    }

    return version, state_map, zone_table

# ============================
# Runtime helpers
# ============================
def _tier_for_value(v: float, breaks: List[float], labels: List[str]) -> str:
    for i in range(len(breaks)-1):
        if breaks[i] <= v < breaks[i+1]:
            return labels[i]
    return labels[-1]

def _mult_lookup_ab(version: dict, zone: str, tier: str) -> Tuple[float,float]:
    # Build cache on first call
    if "_mult_index_ab" not in version:
        idx = {}
        for r in version.get("zt_multipliers", []):
            if "a" in r and "b" in r:
                idx[(r["zone"], r["tier"])] = (float(r["a"]), float(r["b"]))
            elif "multiplier" in r:  # backward compatibility
                idx[(r["zone"], r["tier"])] = (float(r["multiplier"]), 0.0)
        version["_mult_index_ab"] = idx
    return version["_mult_index_ab"].get(
        (zone, tier),
        (1.0 + version.get("params", {}).get("target_mean", 0.18), 0.0)
    )

def _tier_norm_lookup(version: dict, tier: str) -> Tuple[float,float]:
    if "_tier_norm_index" not in version:
        version["_tier_norm_index"] = {r["tier"]: (float(r["mid"]), float(r["width"]))
                                       for r in version.get("tier_norm", [])}
    return version["_tier_norm_index"].get(tier, (0.0, 1.0))

def _state_adjust_lookup(version: dict, state: str) -> float:
    if "_state_adjust_index" not in version:
        version["_state_adjust_index"] = {r["state"]: float(r["c"])
                                          for r in version.get("state_adjust", [])}
    return version["_state_adjust_index"].get(state, 0.0)

def price_for(msrp: float, state: str, version: dict) -> float:
    """Compute price for a single (MSRP, state) using the learned model + optional guard."""
    state = str(state).strip().upper()

    # map state -> zone
    if "_state_index" not in version:
        version["_state_index"] = {r["state"]: r["zone"] for r in version.get("state_map", [])}
    zone = version["_state_index"].get(state, version["zones"][0]["zone"])

    # stabilized zone expected ratio
    zrat = {r["zone"]: float(r["expected_cost_ratio"]) for r in version.get("zones", [])}.get(zone, 1.0)

    # tier
    breaks = version["tiers"]["breaks"]
    labels = version["tiers"]["labels"]
    tier = _tier_for_value(float(msrp), breaks, labels)

    # tier normalization
    mid, width = _tier_norm_lookup(version, tier)
    x = (float(msrp) - float(mid)) / max(1.0, float(width))

    # multipliers
    a, b = _mult_lookup_ab(version, zone, tier)
    c = _state_adjust_lookup(version, state)
    m = max(0.2, min(5.0, a + b*x + c))

    price = float(msrp) * float(zrat) * float(m)

    # Optional runtime guard
    guard = version.get("guard")
    if guard and guard.get("low") is not None and guard.get("high") is not None:
        # We need cost to enforce guard precisely. If unavailable at runtime, you can skip this clip.
        # If you pass cost in, use price_for_with_cost below.
        pass

    return price

def price_for_with_cost(msrp: float, cost_to_ulp: float, state: str, version: dict) -> float:
    """Same as price_for, but applies guard exactly since cost is provided."""
    p = price_for(msrp, state, version)
    guard = version.get("guard")
    if guard and guard.get("low") is not None and guard.get("high") is not None:
        lo = cost_to_ulp * (1.0 + float(guard["low"]))
        hi = cost_to_ulp * (1.0 + float(guard["high"]))
        p = min(max(p, lo), hi)
    return p

def price_for_many(msrps: List[float], states: List[str], version: dict) -> List[float]:
    return [price_for(msrp, state, version) for msrp, state in zip(msrps, states)]
