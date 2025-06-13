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
