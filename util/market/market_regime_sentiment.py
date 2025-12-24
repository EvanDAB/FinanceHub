import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

REGIME_LABELS = [
    "Very Bearish",
    "Bearish",
    "Neutral",
    "Bullish",
    "Very Bullish",
]

def get_spy_vix_regimes(
    spy_close: pd.Series,
    vix_index: pd.Series,
    n_states: int = 5,
    random_state: int = 42
):
    """
    Fit a Gaussian HMM using SPY returns and VIX returns to classify market regimes.

    Parameters
    ----------
    spy_close : pd.Series
        SPY (or S&P 500) closing prices, DateTimeIndex.
    vix_index : pd.Series
        VIX index values (e.g. from FRED), DateTimeIndex.
    n_states : int
        Number of HMM states.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with SPY, VIX, returns, HMM state and regime label.
    current_regime : str
        Regime for the latest date.
    hmm_model : GaussianHMM
        Trained HMM model.
    state_order : np.ndarray
        States ordered from most bearish (0) to most bullish (last).
    """

    # 1) Align SPY & VIX on the same dates
    # Convert to Series if DataFrame, then rename
    if isinstance(spy_close, pd.DataFrame):
        spy_series = spy_close.squeeze()
    else:
        spy_series = spy_close
    
    if isinstance(vix_index, pd.DataFrame):
        vix_series = vix_index.squeeze()
    else:
        vix_series = vix_index
    
    df = pd.concat(
        [
            spy_series.rename("spy"),
            vix_series.rename("vix"),
        ],
        axis=1
    ).sort_index()

    # Forward fill VIX if some SPY days don't have VIX value
    df["vix"] = df["vix"].ffill()

    # Drop any days missing either
    df = df.dropna()

    # 2) Compute log returns
    df["spy_ret"] = np.log(df["spy"] / df["spy"].shift(1))
    df["vix_ret"] = np.log(df["vix"] / df["vix"].shift(1))

    df = df.dropna()  # drop the first NaN return

    # 3) Build feature matrix for HMM: SPY return + VIX return
    X = df[["spy_ret", "vix_ret"]].values

    # 4) Fit HMM
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=random_state,
    )
    hmm.fit(X)

    # 5) Decode most likely state sequence
    hidden_states = hmm.predict(X)
    df["state"] = hidden_states

    # 6) Order states by average SPY return (bearish -> bullish)
    state_means_spy = []
    for s in range(n_states):
        state_returns = df.loc[df["state"] == s, "spy_ret"]
        mean_ret = state_returns.mean() if len(state_returns) > 0 else np.nan
        state_means_spy.append(mean_ret)

    state_means_spy = np.array(state_means_spy)
    state_order = np.argsort(state_means_spy)  # lowest -> highest

    # 7) Map HMM states to regime labels
    state_to_regime = {}
    for rank, state_idx in enumerate(state_order):
        label_idx = min(rank, len(REGIME_LABELS) - 1)
        state_to_regime[state_idx] = REGIME_LABELS[label_idx]

    df["regime"] = df["state"].map(state_to_regime)

    # 8) Current regime
    current_regime = df["regime"].iloc[-1]

    return df, current_regime, hmm, state_order