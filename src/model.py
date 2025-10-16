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
    target_mean: float = 0.18
    band_low: float = 0.10
    band_high: float = 0.40
    band_target: float = 0.80
    zones: int = 4
    msrp_breaks: Tuple[float, ...] = (0.0, 500.0, 1000.0, 2000.0, float("inf"))
    shrinkage: float = 0.80
    iters: int = 120
    lr_mean: float = 0.30
    lr_tail: float = 0.15
    change_cap_pct: float = 0.07

# ----------------------------
# Loader (handles both paths and uploads)
# ----------------------------
def _pick_column(cols: List[str], guess, fallback_idx: int):
    """Pick a column matching 'guess' (case-insensitive substring), else fallback safely."""
    try:
        g = str(guess).lower()
    except Exception:
        g = ""
    matches = [c for c in cols if g in str(c).lower()]
    if matches:
        return matches[0]
    if not cols:
        raise KeyError("No columns found in uploaded file.")
    # clamp fallback index into range
    return cols[min(max(fallback_idx, 0), len(cols) - 1)]

def load_dataframe(source,
                   msrp_col: Optional[str] = None,
                   cost_col: Optional[str] = None,
                   state_col: Optional[str] = None) -> pd.DataFrame:
    """
    Load CSV/XLSX from either a filesystem path (str) OR a file-like object
    (e.g., Streamlit's UploadedFile). Maps to MSRP, Cost_to_ULP, Destination_State.
    """
    # --- Read file into DataFrame ---
    if isinstance(source, str):
        name = source.lower()
        if name.endswith(".csv"):
            src = pd.read_csv(source)
        else:
            src = pd.read_excel(source)
    else:
        # Streamlit UploadedFile or BytesIO
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

    # Validate explicit selections if provided
    missing = []
    if msrp_col and msrp_col not in cols:   missing.append(f"MSRP='{msrp_col}'")
    if cost_col and cost_col not in cols:   missing.append(f"Cost to ULP='{cost_col}'")
    if state_col and state_col not in cols: missing.append(f"Destination State='{state_col}'")
    if missing:
        raise KeyError(f"Missing columns: {', '.join(missing)}. Found: {cols}")

    # Heuristic fallbacks if not provided
    msrp_use  = msrp_col  or _pick_column(cols, "msrp", 0)
    cost_use  = cost_col  or _pick_column(cols, "cost to ulp", 1 if len(cols) > 1 else 0)
    state_use = state_col or _pick_column(cols, "destination state", 2 if len(cols) > 2 else 0)

    df = src.rename(columns={
        msrp_use: "MSRP",
        cost_use: "Cost_to_ULP",
        state_use: "Destination_State"
    }).copy()

    # Clean types
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
def _assign_zones(df: pd.DataFrame, zones: int):
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
    zone_ratio = (
        state_stats.groupby("Zone")
        .apply(lambda g: np.average(g["Median_Cost_Ratio"], weights=g["Count"]))
        .rename("Zone_Expected_Cost_Ratio")
        .reset_index()
    )
    return state_stats, zone_ratio

def _assign_tiers(msrp: pd.Series, breaks: Tuple[float, ...]):
    bins = list(breaks)
    labels = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        labels.append(f"{int(lo)}–{('∞' if hi==float('inf') else int(hi))}")
    idx = pd.cut(msrp, bins=bins, right=False, labels=labels, include_lowest=True)
    return idx.astype(str), labels, bins

# ----------------------------
# Helpers
# ----------------------------
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

# ----------------------------
# Core evaluation
# ----------------------------
def _evaluate(dfz, mults_ab, ratio_col):
    key = list(zip(dfz["Zone"], dfz["Tier"]))
    a = np.array([mults_ab.get(k, (1.0, 0.0))[0] for k in key], float)
    b = np.array([mults_ab.get(k, (1.0, 0.0))[1] for k in key], float)
    m = np.clip(a + b * dfz["MSRP_Norm"].values, 0.2, 5.0)
    price = dfz["MSRP"].values * dfz[ratio_col].values * m
    margin = (price - dfz["Cost_to_ULP"].values) / dfz["Cost_to_ULP"].values
    return pd.Series(price, index=dfz.index), pd.Series(margin, index=dfz.index)

# ----------------------------
# Build zones, tiers, and normalization
# ----------------------------
def build_zones_and_tiers(df, params):
    state_stats, zone_ratio = _assign_zones(df, params.zones)
    global_ratio = df["Cost_to_ULP"].sum() / df["MSRP"].sum()
    dfz = df.merge(state_stats[["Destination_State", "Zone"]], on="Destination_State", how="left")
    dfz = dfz.merge(zone_ratio, on="Zone", how="left")
    dfz["Adj_Zone_Ratio"] = params.shrinkage * dfz["Zone_Expected_Cost_Ratio"] + (1 - params.shrinkage) * global_ratio

    tier_series, tier_labels, breaks = _assign_tiers(dfz["MSRP"], params.msrp_breaks)
    dfz["Tier"] = tier_series

    rows = []
    for lab in tier_labels:
        g = dfz[dfz["Tier"] == lab]
        if g.empty:
            mid, width = 0.0, 1.0
        else:
            w = g["Cost_to_ULP"].values
            v = g["MSRP"].values
            mid = _weighted_quantile(v, w, 0.5)
            p10 = _weighted_quantile(v, w, 0.1)
            p90 = _weighted_quantile(v, w, 0.9)
            width = max(1.0, p90 - p10)
        rows.append({"tier": lab, "mid": float(mid), "width": float(width)})
    tnorm = pd.DataFrame(rows)
    mid_map = dict(zip(tnorm["tier"], tnorm["mid"]))
    wid_map = dict(zip(tnorm["tier"], tnorm["width"]))
    dfz["Tier_Mid"] = dfz["Tier"].map(mid_map)
    dfz["Tier_Width"] = dfz["Tier"].map(wid_map)
    dfz["MSRP_Norm"] = (dfz["MSRP"] - dfz["Tier_Mid"]) / dfz["Tier_Width"]
    return state_stats, zone_ratio, dfz, global_ratio, tnorm

# ----------------------------
# Calibration (a,b)
# ----------------------------
def calibrate(dfz, params, prev_multipliers=None):
    keys = sorted(set(zip(dfz["Zone"], dfz["Tier"])))
    mults = {k: (1.0 + params.target_mean, 0.0) for k in keys}
    w = np.where(dfz["Cost_to_ULP"] > 0, dfz["Cost_to_ULP"], 1.0)
    ratio_col = "Adj_Zone_Ratio"

    for _ in range(params.iters):
        price, marg = _evaluate(dfz, mults, ratio_col)
        target_rev = dfz["Cost_to_ULP"].sum() * (1 + params.target_mean)
        cur_rev = float(price.sum())
        if cur_rev != 0:
            scale = target_rev / cur_rev
            step = 1.0 + params.lr_mean * (scale - 1.0)
            for k in mults:
                a, b = mults[k]
                mults[k] = (a * step, b * step)

        ztdf = pd.DataFrame({
            "Zone": dfz["Zone"], "Tier": dfz["Tier"],
            "marg": marg, "w": w, "x": dfz["MSRP_Norm"]
        })
        for (z,t), g in ztdf.groupby(["Zone","Tier"]):
            k = (z,t)
            a,b = mults[k]
            totw = float(g["w"].sum())
            if totw <= 0: 
                mults[k] = (a,b)
                continue
            below_w = float(g.loc[g["marg"] < params.band_low, "w"].sum())
            above_w = float(g.loc[g["marg"] > params.band_high, "w"].sum())
            inside_w = totw - below_w - above_w
            inside_r = inside_w / totw
            if inside_r < params.band_target:
                # shift center against heavier tail
                skew = (below_w - above_w) / totw
                a *= (1.0 + params.lr_tail * skew)
                # tilt with slope via correlation to "outside" direction
                outsig = np.zeros(len(g))
                outsig[g["marg"] < params.band_low] = +1.0
                outsig[g["marg"] > params.band_high] = -1.0
                x = g["x"].values
                wgt = g["w"].values
                wx = np.average(x, weights=wgt)
                wo = np.average(outsig, weights=wgt)
                cov = np.average((x - wx) * (outsig - wo), weights=wgt)
                b += params.lr_tail * 0.5 * float(cov)
                b = max(-0.5, min(0.5, b))
            mults[k] = (a,b)

    if prev_multipliers:
        capped = {}
        for k, (a,b) in mults.items():
            prev_a, prev_b = prev_multipliers.get(k, (a,b))
            hi_a, lo_a = prev_a*(1+params.change_cap_pct), prev_a*(1-params.change_cap_pct)
            hi_b, lo_b = prev_b*(1+params.change_cap_pct), prev_b*(1-params.change_cap_pct)
            capped[k] = (max(lo_a,min(hi_a,a)), max(lo_b,min(hi_b,b)))
        mults = capped
    return mults

def normalize_total(dfz, mults, params):
    price,_ = _evaluate(dfz, mults, "Adj_Zone_Ratio")
    target_rev = dfz["Cost_to_ULP"].sum() * (1+params.target_mean)
    scale = float(target_rev / price.sum()) if price.sum() else 1.0
    return {k: (a*scale, b*scale) for k,(a,b) in mults.items()}

# ----------------------------
# Scoring
# ----------------------------
def score(dfz, mults, params):
    price, marg = _evaluate(dfz, mults, "Adj_Zone_Ratio")
    total_cost = float(dfz["Cost_to_ULP"].sum())
    total_rev = float(price.sum())
    mean_w = (total_rev / total_cost - 1.0) if total_cost > 0 else float("nan")
    w = np.where(dfz["Cost_to_ULP"] > 0, dfz["Cost_to_ULP"], 1.0)
    wsum = w.sum() or 1.0
    inside_w = float(w[(marg.between(params.band_low, params.band_high))].sum()/wsum)
    below_w = float(w[marg < params.band_low].sum()/wsum)
    above_w = float(w[marg > params.band_high].sum()/wsum)
    return {
        "total_cost": total_cost,
        "total_revenue": total_rev,
        "mean_margin": mean_w,
        "pct_inside": inside_w,
        "pct_below": below_w,
        "pct_above": above_w,
    }

# ----------------------------
# Build version payload
# ----------------------------
def build_version(df, params, prev=None):
    state_stats, zone_ratio, dfz, global_ratio, tnorm = build_zones_and_tiers(df, params)
    prev_mults = None
    if prev and "zt_multipliers" in prev:
        prev_mults = {}
        for r in prev["zt_multipliers"]:
            if "a" in r and "b" in r:
                prev_mults[(r["zone"],r["tier"])] = (float(r["a"]), float(r["b"]))
            else:
                prev_mults[(r["zone"],r["tier"])] = (float(r["multiplier"]), 0.0)
    mults = calibrate(dfz, params, prev_mults)
    mults = normalize_total(dfz, mults, params)
    metrics = score(dfz, mults, params)

    zone_table = dfz.groupby("Zone").apply(
        lambda g: pd.Series({"Zone_Expected_Cost_Ratio": float(np.average(g["Adj_Zone_Ratio"], weights=g["Cost_to_ULP"]))})
    ).reset_index()

    state_map = state_stats[["Destination_State","Zone"]].copy()

    zt_rows = [{"zone":z,"tier":t,"a":float(a),"b":float(b)} for (z,t),(a,b) in mults.items()]

    version = {
        "params": vars(params),
        "metrics": metrics,
        "zones": [{"zone":r["Zone"],"expected_cost_ratio":r["Zone_Expected_Cost_Ratio"]} for _,r in zone_table.iterrows()],
        "state_map":[{"state":r["Destination_State"],"zone":r["Zone"]} for _,r in state_map.iterrows()],
        "tiers":{"breaks":list(params.msrp_breaks),
                 "labels":[str(x) for x in tnorm["tier"].tolist()]},
        "zt_multipliers":zt_rows,
        "tier_norm":[{"tier":r["tier"],"mid":float(r["mid"]),"width":float(r["width"])} for _,r in tnorm.iterrows()]
    }
    return version, state_map, zone_table

# ----------------------------
# Runtime helpers
# ----------------------------
def _tier_for_value(v, breaks, labels):
    for i in range(len(breaks)-1):
        if breaks[i] <= v < breaks[i+1]:
            return labels[i]
    return labels[-1]

def _mult_lookup_ab(version, zone, tier):
    if "_mult_index_ab" not in version:
        version["_mult_index_ab"] = {(r["zone"],r["tier"]):(r.get("a",r.get("multiplier",1.0)), r.get("b",0.0))
                                     for r in version.get("zt_multipliers",[])}
    return version["_mult_index_ab"].get((zone,tier),(1.0 + version["params"].get("target_mean", 0.18), 0.0))

def _tier_norm_lookup(version,tier):
    if "_tier_norm_index" not in version:
        version["_tier_norm_index"] = {r["tier"]:(r["mid"],r["width"]) for r in version.get("tier_norm",[])}
    return version["_tier_norm_index"].get(tier,(0.0,1.0))

def price_for(msrp,state,version):
    state=str(state).strip().upper()
    smap=version.get("_state_index")
    if smap is None:
        smap={r["state"]:r["zone"] for r in version.get("state_map",[])}
        version["_state_index"]=smap
    zone=smap.get(state,version["zones"][0]["zone"])
    zrat={r["zone"]:r["expected_cost_ratio"] for r in version.get("zones",[])}.get(zone,1.0)
    breaks=version["tiers"]["breaks"]; labels=version["tiers"]["labels"]
    tier=_tier_for_value(float(msrp),breaks,labels)
    mid,width=_tier_norm_lookup(version,tier)
    x=(float(msrp)-float(mid))/max(1.0,float(width))
    a,b=_mult_lookup_ab(version,zone,tier)
    m=max(0.2,min(5.0,a+b*x))
    return float(msrp)*float(zrat)*float(m)

def price_for_many(msrps,states,version):
    return [price_for(m,s,version) for m,s in zip(msrps,states)]
