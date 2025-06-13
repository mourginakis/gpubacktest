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
# gdf.info(memory_usage="deep")


#%% ==================== Get GPU working ====================
import timeit


def gpu_sum():
    return gdf["Close"].sum()

def cpu_sum(): 
    return df["Close"].sum()


gpu_result = gpu_sum()
cpu_result = cpu_sum()

assert (gpu_result - cpu_result) < 1e-6, "GPU and CPU results do not match"

print(f"GPU result: {gpu_result}")
print(f"CPU result: {cpu_result}")


cpu_time = timeit.timeit(cpu_sum, number=1_000)
gpu_time = timeit.timeit(gpu_sum, number=1_000)

print(f"CPU: 100 runs took {cpu_time}, average {cpu_time/100} seconds per run")
print(f"GPU: 100 runs took {gpu_time}, average {gpu_time/100} seconds per run")

