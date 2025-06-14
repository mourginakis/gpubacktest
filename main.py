#%% ==================== Imports ====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cudf, cupy, numpy as np, os

# ---------- smoke test ----------
print(f"cuDF version: {cudf.__version__}")
print(f"GPU: {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
gdf = cudf.DataFrame({"x": cupy.arange(10)})
print(gdf.sum())


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


#%% ==================== Get GPU working ====================
import timeit

# you can do fp32 or fp64. fp32 doesnt seem to have much of 
# a speedup because apparently we're memory bound?
# fp64 is great because we don't sacrifice any accuracy

df64  = df["Close"].astype(np.float64)
gdf64 = gdf["Close"].astype(np.float64)

def cpu_sum(): return df64.sum()
def gpu_sum(): return gdf64.sum()
cpu_result = cpu_sum()
gpu_result = gpu_sum()

assert abs(gpu_result - cpu_result) < 0.1, "GPU and CPU results do not match"

print(f"CPU result: {cpu_result}")
print(f"GPU result: {gpu_result}")

cpu_time = timeit.timeit(cpu_sum, number=1_000)
gpu_time = timeit.timeit(gpu_sum, number=1_000)

print(f"CPU: 1000 runs took {cpu_time}, average {cpu_time/1000} seconds per run")
print(f"GPU: 1000 runs took {gpu_time}, average {gpu_time/1000} seconds per run")


#%% ==================== Buy-and-Hold Backtest ====================


# ---------- example using pct_change ----------
df1 = df.copy()
df1["return"] = df["Close"].pct_change()
df1["return"] = df1["return"].fillna(0)
df1["return_cum"] = (1 + df1["return"]).cumprod() - 1
print(f"Buy-and-Hold Total return: {df1['return_cum'].iloc[-1]:.2%}")


# ---------- example using division ----------
# simple return, exactly equivalent to pct_change
df1 = df.copy()
df1["return"] = df1["Close"] / df1["Close"].shift(1) - 1
df1["return"] = df1["return"].fillna(0)
df1["return_cum"] = (1 + df1["return"]).cumprod() - 1
print(f"Buy-and-Hold Total return: {df1['return_cum'].iloc[-1]:.2%}")


# ---------- example using diff (log return) ----------
df1 = df.copy()
df1["log_return"]      = np.log(df["Close"]).diff()
df1["log_return"]      = df1["log_return"].fillna(0)
df1["log_return_cum"]  = df1["log_return"].cumsum()
df1["return_cum"]      = np.exp(df1["log_return_cum"]) - 1
print(f"Buy-and-Hold Total return (via log returns): {df1['return_cum'].iloc[-1]:.2%}")


# log return allows you to do cumulative add, which is 
# parallelized easily on GPU

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



#%% ==================== SMA Cross ===================

# ultra-readable naive python loop
# source of truth for calulation correctness
# baseline for performance comparison

# TODO: switch to pure integer window sizes?

df["sma_2d"]  = df["Close"].rolling(window='2d').mean()
df["sma_10d"] = df["Close"].rolling(window='10d').mean()

# trim the first 30 days to assure smas are fully populated
df1 = df.loc["2017-02-01":].copy()

# calculate the position target (this is the SMA cross)
# CRITICAL: shift signal down by 1 to eliminate lookahead bias
df1["target"] = (df1['sma_2d'] > df1['sma_10d']).astype(int)
df1['signal'] = df1['target'].diff().shift(1) # 1 or -1 for buy/sell signal

# TODO: fill NaN values in signal with 0 (no action)
df1.head()


#%% ===================== Naive Backtest ====================

def backtest_naive(df: pd.DataFrame) -> pd.DataFrame:
    """naive backtest using an iterative loop"""

    # ---------- init portfolio (0.1% trading fee) ----------
    xfee       = 0.001
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


# run, print
results_naive = backtest_naive(df1)
print(f"Naive Backtest Final Equity: {results_naive['equity'].iloc[-1]:.2f}")



#%% ==================== Vectorized Backtest ====================

def backtest_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    pass


#%% ==================== SMA Cross (Vectorized) ====================

# generate SMA for both periods
df['sma_2d']   = df['Close'].rolling(window='2d').mean()
df['sma_10d']  = df['Close'].rolling(window='10d').mean()
df["return"]   = df["Close"].pct_change()
df["return"]   = df["return"].fillna(0)

df1 = df.loc["2017-02-01":].copy()

# generate signals (1=long, 0=flat) based on crossover
df1['signal'] = (df1['sma_2d'] > df1['sma_10d']).astype(int)
# align to avoid lookahead
df1['position'] = df1['signal'].shift(1).fillna(0)
# apply transaction costs per trade
fee_rate = 0.001  # 0.1% commission per trade
df1['trade'] = df1['position'].diff().abs()
# compute strategy returns net of fees
df1['strategy_return_sma'] = df1['position'] * df1['return']
df1['strategy_return_sma'] = df1['strategy_return_sma'] - df1['trade'] * fee_rate
# compute cumulative returns
df1['cum_return_sma'] = (1 + df1['strategy_return_sma']).cumprod() - 1
print(f"SMA Crossover Total Return (fast=2D, slow=10D): {df1['cum_return_sma'].iloc[-1]:.2%}")


# %%
