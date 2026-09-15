"""
features/engineering.py
========================
Feature engineering pipeline (Section A4.5 equivalent).

Produces a wide feature matrix from the raw OHLC + macro panel:
  - Return / trend features (multi-horizon momentum, MA distances)
  - Volatility features (realized vol, vol-of-vol, VIX level & term proxy)
  - Cap-segment features (mid/small vs large relative performance & z-score)
  - Flow features (FII/DII z-scores, SIP momentum, flow balance ratio)
  - Macro features (real-rate proxy, INR momentum, gilt curve proxy)
  - Cross-asset breadth proxy (fraction of the 3 indices above their 50D MA)

TDA (persistence landscapes) and sector-GCN embeddings from Section A7 are
stubbed out (`add_topological_features`, `add_gnn_features`) with a
documented interface — they require rolling correlation persistent
homology / sector graph data this sandbox does not have populated, but
the feature matrix contract (fixed-width numeric columns, same index as
`df`) is what any real implementation must satisfy to plug in cleanly.
"""
import numpy as np
import pandas as pd


def _zscore(s, window=252):
    return (s - s.rolling(window).mean()) / s.rolling(window).std()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().set_index("date")
    px = out["nifty_close"]
    ret = px.pct_change()

    # --- Return / trend ---
    for h in [1, 5, 21, 63, 126, 252]:
        out[f"ret_{h}d"] = px.pct_change(h)
    out["ma20_dist"] = px / px.rolling(20).mean() - 1
    out["ma50_dist"] = px / px.rolling(50).mean() - 1
    out["ma200_dist"] = px / px.rolling(200).mean() - 1

    # --- Volatility ---
    out["rvol_21d"] = ret.rolling(21).std() * np.sqrt(252)
    out["rvol_63d"] = ret.rolling(63).std() * np.sqrt(252)
    out["vol_of_vol"] = out["rvol_21d"].rolling(21).std()
    out["vix_level"] = out["india_vix"]
    out["vix_z"] = _zscore(out["india_vix"])
    out["vix_term_proxy"] = out["india_vix"] - out["india_vix"].rolling(63).mean()

    # --- Cap-segment ---
    mid_ret = out["midcap_close"].pct_change()
    small_ret = out["smallcap_close"].pct_change()
    out["mid_minus_large_63d"] = (out["midcap_close"].pct_change(63)
                                   - px.pct_change(63))
    out["small_minus_large_63d"] = (out["smallcap_close"].pct_change(63)
                                     - px.pct_change(63))
    out["midcap_valuation_z"] = _zscore(out["midcap_close"] / out["nifty_close"])
    out["breadth_proxy"] = (
        (px > px.rolling(50).mean()).astype(int)
        + (out["midcap_close"] > out["midcap_close"].rolling(50).mean()).astype(int)
        + (out["smallcap_close"] > out["smallcap_close"].rolling(50).mean()).astype(int)
    ) / 3.0

    # --- Flow features ---
    out["fii_z"] = _zscore(out["fii_net_cr"], 63)
    out["dii_z"] = _zscore(out["dii_net_cr"], 63)
    out["flow_balance_ratio"] = out["dii_net_cr"] / (out["fii_net_cr"].abs() + 1e-6)
    out["sip_momentum"] = out["sip_total_cr"].pct_change(21)
    out["fii_cum_21d"] = out["fii_net_cr"].rolling(21).sum()

    # --- Macro ---
    out["usdinr_mom_21d"] = out["usdinr"].pct_change(21)
    out["gilt_level"] = out["gilt10y"]
    out["gilt_chg_63d"] = out["gilt10y"].diff(63)
    out["real_rate_proxy"] = out["gilt10y"] - (out["ret_252d"].rolling(63).mean() * 100)

    return out


def add_topological_features(feat_df: pd.DataFrame, window=60) -> pd.DataFrame:
    """
    Stub: persistence-landscape features from rolling correlation matrices
    (Section A7.2). Requires `gtda` (giotto-tda) + a multi-asset correlation
    tensor not populated in this sandbox's single-index synthetic panel.
    Returns the frame unchanged with a placeholder column so downstream
    code can be written against the final schema now and back-filled later.
    """
    feat_df = feat_df.copy()
    feat_df["tda_persistence_entropy"] = np.nan  # TODO: wire gtda.homology.VietorisRipsPersistence
    return feat_df


def add_gnn_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stub: sector-GCN embeddings (Section A7.3). Requires a sector
    constituent graph + per-stock return panel this sandbox does not have.
    """
    feat_df = feat_df.copy()
    feat_df["gnn_sector_embed_0"] = np.nan  # TODO: wire torch_geometric sector graph
    return feat_df


FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "ma20_dist", "ma50_dist", "ma200_dist",
    "rvol_21d", "rvol_63d", "vol_of_vol", "vix_level", "vix_z", "vix_term_proxy",
    "mid_minus_large_63d", "small_minus_large_63d", "midcap_valuation_z", "breadth_proxy",
    "fii_z", "dii_z", "flow_balance_ratio", "sip_momentum", "fii_cum_21d",
    "usdinr_mom_21d", "gilt_level", "gilt_chg_63d", "real_rate_proxy",
]

if __name__ == "__main__":
    from data.loader import load_market_data
    df = load_market_data()
    feat = build_features(df)
    feat = add_topological_features(feat)
    feat = add_gnn_features(feat)
    print(feat[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]])
    print("n features:", len(FEATURE_COLUMNS))
