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
    shrinkage: float = 0.85  # blend zone ratio with global
    iters: int = 80
    lr_mean: float = 0.25
    lr_tail: float = 0.12
    change_cap_pct: float = 0.05  # optional cap vs previous version

def _pick_column(cols, guess, fallback_idx):
    m = [c for c in cols if guess.lower() in str(c).lower()]
    return m[0] if m else cols[fallback_idx]

def load_dataframe(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        src = pd.read_csv(path)
    else:
        src = pd.read_excel(path)
    msrp_col  = _pick_column(src.columns, "msrp", 1)
    cost_col  = _pick_column(src.columns, "cost to ulp", 3)
    state_col = _pick_column(src.columns, "destination state", 4)
    df = src.rename(columns={
        msrp_col: "MSRP",
        cost_col: "Cost_to_ULP",
        state_col: "Destination_State"
    }).copy()
    df["MSRP"] = pd.to_numeric(df["MSRP"], errors="coerce")
    df["Cost_to_ULP"] = pd.to_numeric(df["Cost_to_ULP"], errors="coerce")
    df["Destination_State"] = df["Destination_State"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["MSRP", "Cost_to_ULP", "Destination_State"])
    df = df[(df["MSRP"] > 0) & (df["Cost_to_ULP"] > 0)].copy()
    return df

def build_zones(df: pd.DataFrame, zones: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # state medians of cost/MSRP
    state_stats = (
        df.assign(cost_ratio=df["Cost_to_ULP"]/df["MSRP"])
          .groupby("Destination_State")
          .agg(Count=("Destination_State", "size"),
               Median_Cost_Ratio=("cost_ratio", "median"))
          .reset_index()
    )
    # quantile cutpoints
    qcuts = np.linspace(0, 100, zones + 1)[1:-1]
    edges = np.percentile(state_stats["Median_Cost_Ratio"], qcuts) if len(state_stats) else []
    labels = ZONES_DEFAULT[:zones]
    def to_zone(v):
        idx = np.searchsorted(edges, v, side="right")
        return labels[idx]
    state_stats["Zone"] = state_stats["Median_Cost_Ratio"].apply(to_zone)

    # expected cost ratio per zone (count-weighted)
    zone_ratio = (
        state_stats.groupby("Zone")
        .apply(lambda g: np.average(g["Median_Cost_Ratio"], weights=g["Count"]))
        .rename("Zone_Expected_Cost_Ratio")
        .reset_index()
    )

    # attach to each row
    dfz = df.merge(state_stats[["Destination_State", "Zone"]], on="Destination_State", how="left")
    dfz = dfz.merge(zone_ratio, on="Zone", how="left")
    return state_stats, zone_ratio, dfz

def _evaluate(dfz: pd.DataFrame, mults: Dict[str, float], use_adjusted_ratio=True) -> Tuple[pd.Series, pd.Series]:
    ratio_col = "Adj_Zone_Ratio" if use_adjusted_ratio and "Adj_Zone_Ratio" in dfz.columns else "Zone_Expected_Cost_Ratio"
    price = dfz["MSRP"] * dfz[ratio_col] * dfz["Zone"].map(mults)
    margin = (price - dfz["Cost_to_ULP"]) / dfz["Cost_to_ULP"]
    return price, margin

def calibrate(dfz: pd.DataFrame, params: CalibParams, prev_multipliers: Dict[str, float] | None = None) -> Dict[str, float]:
    zones = dfz["Zone"].dropna().unique().tolist()
    mults = {z: 1.0 + params.target_mean for z in zones}  # init near target

    # slight shrinkage to stabilize extremes
    global_ratio = (dfz["Cost_to_ULP"].sum() / dfz["MSRP"].sum())
    dfz = dfz.copy()
    dfz["Adj_Zone_Ratio"] = params.shrinkage * dfz["Zone_Expected_Cost_Ratio"] + (1 - params.shrinkage) * global_ratio

    for _ in range(params.iters):
        price, marg = _evaluate(dfz, mults, use_adjusted_ratio=True)
        mean = float(marg.mean())
        # 1) global centering to target
        if not math.isclose(mean, params.target_mean, rel_tol=1e-6, abs_tol=1e-6):
            scale = 1.0 + params.lr_mean * (params.target_mean - mean)
            for z in mults:
                mults[z] *= scale
        # 2) per-zone tightening for band coverage
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

    # optional cap on change vs previous
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
    price, _ = _evaluate(dfz, mults, use_adjusted_ratio=True)
    target_rev = dfz["Cost_to_ULP"].sum() * (1 + params.target_mean)
    scale = float(target_rev / price.sum()) if price.sum() else 1.0
    return {z: m * scale for z, m in mults.items()}

def score(dfz: pd.DataFrame, mults: Dict[str, float], params: CalibParams) -> dict:
    price, marg = _evaluate(dfz, mults, use_adjusted_ratio=True)
    inside = ((marg >= params.band_low) & (marg <= params.band_high)).mean()
    return {
        "total_cost": float(dfz["Cost_to_ULP"].sum()),
        "total_revenue": float(price.sum()),
        "mean_margin": float(marg.mean()),
        "pct_inside": float(inside),
        "pct_below": float((marg < params.band_low).mean()),
        "pct_above": float((marg > params.band_high).mean())
    }

def build_version(df: pd.DataFrame, params: CalibParams, prev: dict|None=None) -> dict:
    state_stats, zone_ratio, dfz = build_zones(df, params.zones)
    prev_mults = None
    if prev:
        prev_mults = {r["zone"]: r["multiplier"] for r in prev["zones"]}
    mults = calibrate(dfz, params, prev_mults)
    mults = normalize_total(dfz, mults, params)
    metrics = score(dfz, mults, params)

    # build exportables
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
            {"zone": row["Zone"],
             "expected_cost_ratio": float(row["Zone_Expected_Cost_Ratio"]),
             "multiplier": float(row["Zone_Multiplier"])}
            for _, row in zone_table.iterrows()
        ],
        "state_map": [
            {"state": r["Destination_State"], "zone": r["Zone"]} for _, r in state_stats.iterrows()
        ]
    }
    return version, state_map, zone_table
