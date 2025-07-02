#%% ==================== Imports ====================
from timeit import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cudf, cupy, numpy as np, os

# ---------- verify cupy/cuDF is working ----------
def smoke_test():
    print(f"cuDF version: {cudf.__version__}")
    print(f"GPU: {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    gdf = cudf.DataFrame({"x": cupy.arange(10)})
    assert gdf.sum().sum() == 45, "GPU smoke test failed"
    print("GPU smoke test passed!")
smoke_test()



# %% ==================== Load Data ====================
# ---------- load btc_usd into memory ----------
df = pd.read_csv("btc_usd.csv.xz", compression="xz")
df.set_index("Timestamp", inplace=True)
df.index = pd.to_datetime(df.index, unit="s")
df.info(memory_usage="deep")

# ---------- create pct and log returns ----------
# note: pct_change and division by a shifted value are equivalent.
df["returns_pct"] = df["Close"].pct_change().fillna(0)
df["returns_pct"] = (df["Close"] / df["Close"].shift(1) - 1).fillna(0)
df["returns_log"] = np.log(df["Close"]).diff().fillna(0)


#%% ==================== Buy-and-Hold Backtest ====================

# ── Long-BTC demo ─────────────────────────────────────────────────────────
# Goal: benchmark numeric paths, neglecting trading logic for now. This lets us
# focus entirely on the nature of the computational optimizations we can use,
# without the complexity of a full trading strategy.
#
# Two optional outputs, with two cost profiles:
#   • Full equity curve  → needs a prefix scan (cumprod / cumsum) on GPU/CPU
#   • Final P&L only     → needs one reduction: Σ logR  → exp  → –1
#
# The first route is already extremely fast on the GPU, but the second route 
# skips the scan entirely; a single kernel distils millions of ticks into 
# one scalar, delivering an absolutely massive wall-clock speed-up.
# note: cudf has a bit of an overhead over just cupy.

# ---------- prepare data ----------
df1  = df.copy()
gdf1 = cudf.from_pandas(df1)

# -------------------- SOLUTION: Full Time Series --------------------
# Multiple solutions to generate the full equity curve of cumulative returns.

# 1: naive loop
def longbtc_cpu_loop():
    growth = [1.0]
    for R in df1["returns_pct"]:
        growth.append(growth[-1] * (1 + R))
    return pd.Series(growth[1:], index=df1.index) - 1

# 2. vectorized pandas (cumulative product)
def longbtc_cpu_cumprod():
    return_cum = (1 + df1["returns_pct"]).cumprod() - 1
    return return_cum

# 3. vectorized pandas (logarithmic sum)
def longbtc_cpu_log_cumsum():
    return_cum = np.exp(df1["returns_log"].cumsum()) - 1
    return return_cum

# 4. vectorized cudf (cumulative product)
def longbtc_gpu_cumprod():
    return_cum = (1 + gdf1["returns_pct"]).cumprod() - 1
    return return_cum

# 5. vectorized cudf (logarithmic sum)
def longbtc_gpu_log_cumsum():
    log_r = gdf1["returns_log"].values
    factor = cupy.exp(cupy.cumsum(log_r)) # cudf does not have .exp yet
    return cudf.Series(factor - 1, index=gdf1.index)


# -------------------- SOLUTION: Only Final Return --------------------
# Skip the prefix-scan operation entirely, and just return the final P&L.

# 6. vectorized cudf (ultimate solution)
def longbtc_gpu_ultimate():
    log_r = gdf1["returns_log"].values
    return cupy.exp(cupy.sum(log_r)) - 1


# Series is cumulative RETURN, not cash:
#   0   → no change
#   0.5 → +50 %
#   95  → +9 500 %  (96 × growth)
# Need dollars?   equity = (return + 1) * initial_cash

# ---------- verify all methods give the same result ----------
assert np.allclose([
    longbtc_cpu_cumprod().iloc[-1],  longbtc_cpu_log_cumsum().iloc[-1],
    longbtc_gpu_cumprod().iloc[-1],  longbtc_gpu_log_cumsum().iloc[-1],
    float(longbtc_gpu_ultimate())],
    longbtc_cpu_loop().iloc[-1]), "Mismatch in final cumulative return"
print(f"Long_BTC cumulative return: {float(longbtc_gpu_ultimate()):.3f}x\n\n")



# -------------------- Performance Comparison --------------------
longbtc_cpu_loop_time         = timeit(longbtc_cpu_loop,       number=4)     / 4
longbtc_cpu_cumprod_time      = timeit(longbtc_cpu_cumprod,    number=100)   / 100
longbtc_cpu_log_cumsum_time   = timeit(longbtc_cpu_log_cumsum, number=100)   / 100
longbtc_gpu_cumprod_time      = timeit(longbtc_gpu_cumprod,    number=1_000) / 1_000
longbtc_gpu_log_cumsum_time   = timeit(longbtc_gpu_log_cumsum, number=1_000) / 1_000
longbtc_gpu_ultimate_time     = timeit(longbtc_gpu_ultimate,   number=1_000) / 1_000

print(
    f"| function                    | seconds per run |\n"
    f"|-----------------------------|-----------------|\n"
    f"| longbtc_cpu_loop            | {longbtc_cpu_loop_time:>15.8f} |\n"
    f"| longbtc_cpu_cumprod         | {longbtc_cpu_cumprod_time:>15.8f} |\n"
    f"| longbtc_cpu_log_cumsum      | {longbtc_cpu_log_cumsum_time:>15.8f} |\n"
    f"| longbtc_gpu_cumprod         | {longbtc_gpu_cumprod_time:>15.8f} |\n"
    f"| longbtc_gpu_log_cumsum      | {longbtc_gpu_log_cumsum_time:>15.8f} |\n"
    f"| longbtc_gpu_ultimate        | {longbtc_gpu_ultimate_time:>15.8f} |\n\n"
    f"Max speedup: {longbtc_cpu_loop_time / longbtc_gpu_ultimate_time:.2f}x"
)


# ------ Output (Ryzen 5 3600, RTX 3080Ti):
# | function                    | seconds per run |
# |-----------------------------|-----------------|
# | longbtc_cpu_loop            |      0.93035735 |
# | longbtc_cpu_cumprod         |      0.03322577 |
# | longbtc_cpu_log_cumsum      |      0.04992039 |
# | longbtc_gpu_cumprod         |      0.00420642 |
# | longbtc_gpu_log_cumsum      |      0.00426720 |
# | longbtc_gpu_ultimate        |      0.00033845 |
#
# Max speedup: 2748.91x



#%% ==================== Plotting ====================

def plot_returns(df: pd.DataFrame, title: str):
    raise NotImplementedError()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df["return_cum"], label="Cumulative Return")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True)
    ax.legend()
    # show the plot
    fig.show()

# plot_returns(df1, "Buy-and-Hold (Log Returns)")



#%% ===================== Backtest Naive ====================

def backtest_naive(df: pd.DataFrame) -> pd.DataFrame:
    """naive backtest using an iterative loop"""
    # ultra-readable naive python loop
    # source of truth for calulation correctness
    # baseline for performance comparison

    # ---------- init portfolio (0.1% trading fee) ----------
    xfee       = 0.000
    cash0      = 100
    cash       = [ cash0]
    size       = [   0.0]
    position   = [     0]
    fees       = [   0.0]
    equity     = [ cash0]
    # position = 0
    # n_trades = 0
    # note: we probably don't need to store position

    # ---------- iterate over the df ----------
    for idx, row in df.iterrows():

        verbose = False
        if verbose:
            if int(idx.timestamp() // 60 ) % 100_000 == 0:
                print(f"Processing {idx}...")

        # ---------- snapshot _now ----------
        price_now     = row["Close"]
        cash_now      = cash[-1]
        size_now      = size[-1]
        position_now  = position[-1]
        equity_now    = equity[-1]

        # CASE: buy signal -> buy the asset
        if row["signal"] == 1 and position_now == 0:
            size_delta     = cash_now / (price_now * (1 + xfee))
            fee_paid       = size_delta * price_now * xfee
            cash_final     = 0.0
            size_final     = size_now + size_delta
            position_final = 1
            equity_final   = cash_final + size_final * price_now

        # CASE: sell signal -> sell the asset
        elif row["signal"] == -1 and position_now == 1:
            fee_paid        = size_now * price_now * xfee
            cash_final      = cash_now + size_now * price_now - fee_paid
            size_final      = 0.0
            position_final  = 0
            equity_final    = cash_final + size_final * price_now

        # CASE: hold signal -> do nothing
        else:
            fee_paid        = 0.0
            cash_final      = cash_now
            size_final      = size_now
            position_final  = position_now
            equity_final    = cash_final + size_final * price_now

        # ---------- store results ----------
        fees.append(fee_paid)
        cash.append(cash_final)
        size.append(size_final)
        position.append(position_final)
        equity.append(equity_final)

    # ---------- create a new df with results ----------
    # truncate the seed values
    cash      = cash[1:]
    size      = size[1:]
    position  = position[1:]
    equity    = equity[1:]

    # TODO: calculate cumulative fees?

    results = pd.DataFrame({
        "cash": cash,
        "size": size,
        "position": position,
        "equity": equity
    }, index=df.index)

    return results


#%% ==================== Backtest Vectorized ====================

def backtest_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """vectorised back test, using pandas (CPU)"""
    xfee = 0.0                       # fees disabled for now
    # per-bar P&L
    df["strategy_returns"] = df["target"].shift(1).fillna(0) * df["returns_pct"]

    # trades (0 → 1 or 1 → 0) for future fee logic
    # df["trade"] = df["target"].diff().abs().fillna(0)
    # df["strategy_returns"] -= df["trade"] * xfee

    # equity curve (start with $100)
    df["cum_return"] = (1 + df["strategy_returns"]).cumprod() - 1
    df["equity"]     = 100 * (1 + df["cum_return"])

    return df

#%% ==================== Backtest cudf ====================

def backtest_cudf(df: cudf.DataFrame) -> cudf.DataFrame:
    """vectorised back test, using cudf (GPU)"""
    xfee = 0.0                       # fees disabled for now
    # per-bar P&L
    df["strategy_returns"] = df["target"].shift(1).fillna(0) * df["returns_pct"]

    # trades (0 → 1 or 1 → 0) for future fee logic
    # df["trade"] = df["target"].diff().abs().fillna(0)
    # df["strategy_returns"] -= df["trade"] * xfee

    # equity curve (start with $100)
    df["cum_return"] = (1 + df["strategy_returns"]).cumprod() - 1
    df["equity"]     = 100 * (1 + df["cum_return"])

    return df

def backtest_cudf_ulti(df: cudf.DataFrame) -> cudf.DataFrame:
    pass

    

#%% ==================== SMA Backtest ====================

# ---------- prepare data ----------
# also trim head to remove incomplete SMA values
minutes_1d  = 24 * 60
window_2d   = 2 * minutes_1d
window_10d  = 10 * minutes_1d
df["sma_2d"]  = df["Close"].rolling(window=window_2d, min_periods=window_2d).mean()
df["sma_10d"] = df["Close"].rolling(window=window_10d, min_periods=window_10d).mean()
df1 = df.loc["2017-02-01":].copy()
assert df1.isna().sum().sum() == 0, "data error"

# ---------- compute the strategy ----------
# `target` is 1 for long, 0 for flat, and denotes the desired position
# `signal` is 1 for buy, -1 for sell, and denotes the action needed
# CRITICAL: shift down by 1 to eliminate lookahead bias
df1["target"] = (df1['sma_2d'] > df1['sma_10d']).astype(int).shift(1).fillna(0)
df1["signal"] = df1["target"].diff().fillna(0)

df1.head(10)


#%% ==================== Check Correctness ====================
gdf1 = cudf.DataFrame.from_pandas(df1)

result1 = backtest_naive(df1)
result2 = backtest_vectorized(df1)
result3 = backtest_cudf(gdf1)

result1 = result1["equity"].iloc[-1]
result2 = result2["equity"].iloc[-1]
result3 = result3["equity"].iloc[-1]

# all should equal: 5865.51 tested with 0.000 fee.
print(f"Loop      |  vectorized |   cudf  \n"
      f"{result1:.3f} == {result2:.3f} == {result3:.3f}\n\n")

assert np.allclose([result1, result2], result3), "mismatch in equity results"


#%% ==================== Benchmark Speed ====================


# TODO: implement speed?
# TODO: implement slippage?


def benchmark_naive():
    backtest_naive(df1)

def benchmark_vectorized():
    backtest_vectorized(df1)

def benchmark_cudf():
    backtest_cudf(gdf1)


naive_time        = timeit(benchmark_naive, number=3)           / 3
vectorized_time   = timeit(benchmark_vectorized, number=1_000)  / 1_000
cudf_time         = timeit(benchmark_cudf, number=1_000)        / 1_000


print(f"CPU Backtest (naive):    {naive_time:.8f}          seconds per run")
print(f"CPU multi-core Backtest: {vectorized_time:.8f} seconds per run")
print(f"GPU Backtest:            {cudf_time:.8f}       seconds per run")
print(f"Max speedup: {naive_time / cudf_time:.3f}x\n\n")


# ------ Output (Ryzen 5 3600, RTX 3080Ti):
# CPU Backtest (naive):    117.18413124  seconds per run
# CPU multi-core Backtest: 0.05004053    seconds per run
# GPU Backtest:            0.01131282    seconds per run
# Max speedup: 10358.529x


#%% ==================== Find a winning strategy? ====================

# TODO: test across 10000s of strategies?

# %%
