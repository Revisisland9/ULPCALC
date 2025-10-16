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
    if path.lower().endswith(".csv"):
        src = pd.read_csv(path)
    else:
        src = pd.read_excel(path)

    cols = list(src.columns)
    msrp_use  = msrp_col  or _pick_column(cols, "msrp",  0 if len(cols)>0 else 0)
    cost_use  = cost_col  or _pick_column(cols, "cost to ulp", 1 if len(cols)>1 else 0)
    state_use = state_col or _pick_column(cols, "destination state", 2 if len(cols)>2 else 0)

    df = src.rename(columns={msrp_use:"MSRP", cost_use:"Cost_to_ULP", state_use:"Destination_State"}).copy()
    df["MSRP"] = pd.to_numeric(df["MSRP"], errors="coerce")
    df["Cost_to_ULP"] = pd.to_numeric(df["Cost_to_ULP"], errors="coerce")
    df["Destination_State"] = df["Destination_State"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["MSRP","Cost_to_ULP","Destination_State"])
    df = df[(df["MSRP"]>0) & (df["Cost_to_ULP"]>0)].copy()
    return df

def build_zones(df: pd.DataFrame, zones: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_stats = (
        df.assign(cost_ratio=df["Cost_to_ULP"]/df["MSRP"])
          .groupby("Destination_State")
          .agg(Count=("Destination_State","size"),
               Median_Cost_Ratio=("cost_ratio","median"),
               Cost_Sum=("Cost_to_ULP","sum"))
          .reset_index()
    )
    qcuts = np.linspace(0, 100, zones+1)[1:-1]
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

    dfz = df.merge(state_stats[["Destination_State","Zone"]], on="Destination_State", how="left")
    dfz = dfz.merge(zone_ratio, on="Zone", how="left")
    return state_stats, zone_ratio, dfz

def _evaluate(dfz: pd.DataFrame, mults: Dict[str,float]) -> Tuple[pd.Series,pd.Series]:
    price = dfz["MSRP"] * dfz["Zone_Expected_Cost_Ratio"] * dfz["Zone"].map(mults)
    margin = (price - dfz["Cost_to_ULP"]) / dfz["Cost_to_ULP"]
    return price, margin

def calibrate(dfz: pd.DataFrame, params: CalibParams, prev_multipliers: Dict[str,float] | None = None) -> Dict[str,float]:
    zones = dfz["Zone"].dropna().unique().tolist()
    mults = {z: 1.0 + params.target_mean for z in zones}

    # precompute weights (cost-weighted optimization)
    cost_w = dfz["Cost_to_ULP"].values
    cost_w = np.where(cost_w>0, cost_w, 1.0)

    for _ in range(params.iters):
        price, marg = _evaluate(dfz, mults)

        # 1) Global centering to cost-weighted target
        # cost-weighted mean margin = (Σ(price - cost))/Σ(cost) = (Σprice/Σcost) - 1
        target_rev = dfz["Cost_to_ULP"].sum() * (1 + params.target_mean)
        cur_rev = float(price.sum())
        if cur_rev != 0:
            scale = (target_rev / cur_rev)
            # gentle step toward target to avoid overshoot
            step = 1.0 + params.lr_mean * (scale - 1.0)
            for z in mults:
                mults[z] *= step

        # 2) Per-zone tightening using COST-WEIGHTED band coverage
        zdf = pd.DataFrame({
            "Zone": dfz["Zone"].values,
            "marg": marg.values,
            "w": cost_w
        })
        def wmean(x, w): 
            w = np.asarray(w); x = np.asarray(x)
            return (x*w).sum() / (w.sum() if w.sum()!=0 else 1.0)
        stats = (zdf.groupby("Zone")
                  .apply(lambda g: pd.Series({
                      "mean_w": wmean(g["marg"], g["w"]),
                      "below_w": g.loc[g["marg"] < params.band_low, "w"].sum() / g["w"].sum(),
                      "above_w": g.loc[g["marg"] > params.band_high, "w"].sum() / g["w"].sum(),
                      "inside_w": g.loc[(g["marg"]>=params.band_low)&(g["marg"]<=params.band_high), "w"].sum() / g["w"].sum()
                  }))
                  .reset_index())

        for _, r in stats.iterrows():
            z = r["Zone"]
            if r["inside_w"] < params.band_target:
                if r["below_w"] > 0.05:
                    mults[z] *= (1 + params.lr_tail * min(0.10, r["below_w"]))
                if r["above_w"] > 0.05:
                    mults[z] *= (1 - params.lr_tail * min(0.10, r["above_w"]))

    # optional cap vs previous
    if prev_multipliers:
        capped = {}
        for z,m in mults.items():
            prev = prev_multipliers.get(z, m)
            hi = prev*(1+params.change_cap_pct)
            lo = prev*(1-params.change_cap_pct)
            capped[z] = max(lo, min(hi, m))
        mults = capped
    return mults

def normalize_total(dfz: pd.DataFrame, mults: Dict[str,float], params: CalibParams) -> Dict[str,float]:
    price, _ = _evaluate(dfz, mults)
    target_rev = dfz["Cost_to_ULP"].sum() * (1 + params.target_mean)
    scale = float(target_rev / price.sum()) if price.sum() else 1.0
    return {z: m*scale for z,m in mults.items()}

def score(dfz: pd.DataFrame, mults: Dict[str,float], params: CalibParams) -> dict:
    price, marg = _evaluate(dfz, mults)
    # unweighted KPIs (for reference)
    mean_unw = float(marg.mean())
    pct_inside_unw = float(((marg>=params.band_low)&(marg<=params.band_high)).mean())
    pct_below_unw = float((marg<params.band_low).mean())
    pct_above_unw = float((marg>params.band_high).mean())
    # cost-weighted KPIs (primary)
    w = dfz["Cost_to_ULP"].values
    w = np.where(w>0, w, 1.0)
    mean_w = float(((price.sum()/dfz["Cost_to_ULP"].sum()) - 1.0)) if dfz["Cost_to_ULP"].sum()>0 else mean_unw
    inside_w = float((w[((marg>=params.band_low)&(marg<=params.band_high))]).sum() / w.sum())
    below_w  = float((w[(marg<params.band_low)]).sum() / w.sum())
    above_w  = float((w[(marg>params.band_high)]).sum() / w.sum())

    return {
        "total_cost": float(dfz["Cost_to_ULP"].sum()),
        "total_revenue": float(price.sum()),
        # primary (cost-weighted)
        "mean_margin": mean_w,
        "pct_inside": inside_w,
        "pct_below": below_w,
        "pct_above": above_w,
        # reference (unweighted)
        "mean_margin_unweighted": mean_unw,
        "pct_inside_unweighted": pct_inside_unw,
        "pct_below_unweighted": pct_below_unw,
        "pct_above_unweighted": pct_above_unw
    }

def build_version(df: pd.DataFrame, params: CalibParams, prev: dict|None=None) -> dict:
    state_stats, zone_ratio, dfz = build_zones(df, params.zones)
    prev_mults = {r["zone"]: r["multiplier"] for r in prev.get("zones", [])} if prev else None
    mults = calibrate(dfz, params, prev_mults)
    mults = normalize_total(dfz, mults, params)
    metrics = score(dfz, mults, params)

    zone_table = zone_ratio.copy()
    zone_table["Zone_Multiplier"] = zone_table["Zone"].map(mults)

    state_map = state_stats[["Destination_State","Zone"]].merge(
        zone_table[["Zone","Zone_Expected_Cost_Ratio","Zone_Multiplier"]],
        on="Zone", how="left"
    )

    version = {
        "params": vars(params),
        "metrics": metrics,
        "zones": [
            {"zone": r["Zone"],
             "expected_cost_ratio": float(r["Zone_Expected_Cost_Ratio"]),
             "multiplier": float(r["Zone_Multiplier"])}
            for _, r in zone_table.iterrows()
        ],
        "state_map": [
            {"state": r["Destination_State"], "zone": r["Zone"]}
            for _, r in state_stats.iterrows()
        ]
    }
    return version, state_map, zone_table

