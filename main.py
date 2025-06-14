#%% ==================== Imports ====================
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cudf, cupy, numpy as np, os

def gpu_smoke_test():
    print(f"cuDF version: {cudf.__version__}")
    print(f"GPU: {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    gdf = cudf.DataFrame({"x": cupy.arange(10)})
    assert gdf.sum().sum() == 45, "GPU smoke test failed"
    print("GPU smoke test passed!")

gpu_smoke_test()

# %% ==================== Load Data ====================

# ---------- load data (CPU) ----------
df = pd.read_csv("btc_usd.csv.xz", compression="xz")
df.info(memory_usage="deep")
df.set_index("Timestamp", inplace=True)
df.index = pd.to_datetime(df.index, unit="s")

# ---------- load data (GPU) ----------
# cudf does not support xz, so load from cpu dataframe
gdf = cudf.DataFrame(df)
gdf.info(memory_usage="deep")


#%% ==================== Buy-and-Hold Backtest ====================

# Here, we demonstrate multiple identical long-btc backtests using different 
# methods of computation. This is a great introduction to gpu backtesting 
# because there is no complex fee logic, and the general idea for 
# the computational speedup is easy to understand.
#
# Note: in pandas, pct_change is the same thing as division by a shifted value.
# to put it explicitly:
# df["return1"] = df["Close"].pct_change()
# df["return1"] = df["Close"] / df["Close"].shift(1) - 1
# We will avoid using pct_change to avoid confusion, 
# preferring to be explicit.
#

df1  = df.copy()
gdf1 = gdf.copy()

# 1. simple iteration loop
def long_btc_naive():
    returns = df1["Close"] / df1["Close"].shift(1) - 1
    returns = returns.fillna(0)
    wealth = 1.0
    for R in returns:
        wealth *= (1 + R)
    return wealth - 1

print(f"Buy-and-Hold Total return (naive loop): {long_btc_naive():.2%}")

# 2. using vectorized pandas (cumprod)
def long_btc_vectorized():
    returns = df1["Close"] / df1["Close"].shift(1) - 1
    returns = returns.fillna(0)
    return_cum = (1 + returns).cumprod() - 1
    return return_cum.iloc[-1]

print(f"Buy-and-Hold Total return (vectorized): {long_btc_vectorized():.2%}")

# 3. using vectorized pandas (log->cumsum)
def long_btc_vectorized_log():
    log_returns = np.log(df1["Close"]).diff()
    log_returns = log_returns.fillna(0)
    return_cum = np.exp(log_returns.cumsum()) - 1
    return return_cum.iloc[-1]
print(f"Buy-and-Hold Total return (log returns): {long_btc_vectorized_log():.2%}")

# 4. using vectorized cudf (cumsum) 
def long_btc_cudf():
    returns = gdf1["Close"].pct_change().fillna(0)
    wealth  = (1 + returns).cumprod()
    return float(wealth.iloc[-1] - 1)

# 5 . using cupy (log-diff, cumsum)
def long_btc_cupy():
    # here we stay entirely in cupy so there's no cudf overhead
    # which makes it like 2x as fast
    # 1. pull the Series onto the GPU as a contiguous CuPy array
    # 2. log-diff (fast, numerically stable)
    # 3. cumulative wealth
    px = gdf1["Close"].values          # cupy.ndarray
    log_ret = cupy.diff(cupy.log(px), prepend=cupy.log(px[0]))
    wealth = cupy.exp(cupy.cumsum(log_ret))
    return float(wealth[-1] - 1)


print(f"Buy-and-Hold Total return (cudf): {long_btc_cudf():.2%}")


# what's really cool, is that if you don't need any of the intermediate
# results, you can completely skip the cumsum, and just calculate the
# sum of the log returns, and then exponentiate it to bring it back out
# of log space.

def long_btc_ultimate():
    px = gdf1["Close"].values
    log_ret = cupy.diff(cupy.log(px), prepend=cupy.log(px[0]))  # first diff = 0
    total_log_r = cupy.sum(log_ret)                             # Σ log-returns
    return float(cupy.exp(total_log_r) - 1)                     # (P_T/P_0) – 1


print(f"Buy-and-Hold Total return (ultimate): {long_btc_ultimate():.2%}")


# assert they are all equal
assert abs(long_btc_naive()   - long_btc_vectorized()    ) < 0.01
assert abs(long_btc_naive()   - long_btc_vectorized_log()) < 0.01
assert abs(long_btc_naive()   - long_btc_cudf()          ) < 0.01

long_btc_naive_time      = timeit.timeit(long_btc_naive, number=100)
long_btc_vectorized_time = timeit.timeit(long_btc_vectorized, number=1_000)
long_btc_cudf_time       = timeit.timeit(long_btc_cudf, number=1_000)
long_btc_cupy_time       = timeit.timeit(long_btc_cupy, number=1_000)
long_btc_ultimate_time   = timeit.timeit(long_btc_ultimate, number=1_000)
print(f"Naive loop: 100 runs took {long_btc_naive_time}, "
      f"average {long_btc_naive_time/100} seconds per run")
print(f"Vectorized pandas: 1000 runs took {long_btc_vectorized_time}, "
      f"average {long_btc_vectorized_time/1000} seconds per run")
print(f"Vectorized cudf: 1000 runs took {long_btc_cudf_time}, "
      f"average {long_btc_cudf_time/1000} seconds per run")
print(f"Long-btc cupy: 1000 runs took {long_btc_cupy_time}, "
      f"average {long_btc_cupy_time/1000} seconds per run")
print(f"Ultimate cupy: 1000 runs took {long_btc_ultimate_time}, "
      f"average {long_btc_ultimate_time/1000} seconds per run")



#%% ==================== Plotting ====================

def plot_returns(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df["return_cum"], label="Cumulative Return")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True)
    ax.legend()
    # show the plot
    fig.show()

plot_returns(df1, "Buy-and-Hold (Log Returns)")



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
    df["strategy_returns"] = df["target"].shift(1).fillna(0) * df["returns"]

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
    df["strategy_returns"] = df["target"].shift(1).fillna(0) * df["returns"]

    # trades (0 → 1 or 1 → 0) for future fee logic
    # df["trade"] = df["target"].diff().abs().fillna(0)
    # df["strategy_returns"] -= df["trade"] * xfee

    # equity curve (start with $100)
    df["cum_return"] = (1 + df["strategy_returns"]).cumprod() - 1
    df["equity"]     = 100 * (1 + df["cum_return"])

    return df
    

#%% ==================== SMA Backtest ====================

# ---------- prepare data ----------
# TODO: switch to pure integer window sizes?
df["sma_2d"]  = df["Close"].rolling(window='2d').mean()
df["sma_10d"] = df["Close"].rolling(window='10d').mean()
df["returns"] = df["Close"].pct_change()

# ---------- trim the NaN values ----------
# trim the first 30 days to assure smas are fully 
# populated, and no NaNs
df1 = df.loc["2017-02-01":].copy()

# ---------- run the strategy ----------
# `target` is 1 for long, 0 for flat, and denotes the desired position
# `signal` is 1 for buy, -1 for sell, and denotes the action needed
# CRITICAL: shift down by 1 to eliminate lookahead bias
df1["target"] = (df1['sma_2d'] > df1['sma_10d']).astype(int).shift(1).fillna(0)
df1["signal"] = df1["target"].diff().fillna(0)

df1.head()


#%% ==================== Run the backtests ====================

results_vectorized = backtest_vectorized(df1)
print(f"Vectorized Backtest Final Equity: {results_vectorized['equity'].iloc[-1]:.2f}")

results_naive = backtest_naive(df1)
print(f"Naive Backtest Final Equity: {results_naive['equity'].iloc[-1]:.2f}")

df2 = cudf.DataFrame.from_pandas(df1)
results_cudf = backtest_cudf(df2)
print(f"cuDF Backtest Final Equity: {results_cudf['equity'].iloc[-1]:.2f}")

# both should equal: 5865.51 with 0.000 fee.


#%% ==================== Benchmark Speed ====================
