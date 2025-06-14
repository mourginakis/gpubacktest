#%% ==================== Imports ====================
import pandas as pd


#%% ==================== Load Data ====================
# downloaded data from kaggle (~300mb)
# https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
df = pd.read_csv("btcusd_1-min_data.csv")
print(df.info(memory_usage="deep"), end="\n\n")


#%% ==================== Clean Data ====================

# ---------- keep cols: timestamp, close ----------
# reduces size from 369mb to 107mb
df1 = df[["Timestamp", "Close"]]
print(df1.info(memory_usage="deep"), end="\n\n")

# ---------- truncate from 2017 to 2025 ----------
# full kaggle datset starts from ~2012 
df1.index = pd.to_datetime(df1["Timestamp"], unit='s')
df2 = df1.loc["2017-01-01":"2025-01-01"]
print(df2.info(memory_usage="deep"), end="\n\n")

# ---------- sanity checks, assert clean data ----------
assert df2.isna().sum().sum() == 0, "Data contains NaN values"
assert df2.index.is_monotonic_increasing, "Index not sorted"
assert not df2.index.duplicated().any(), "Duplicate timestamps present"
assert df2.index.equals(
    pd.date_range(df2.index[0], df2.index[-1], freq="1min")
), "Irregular 1-min spacing"



#%% ==================== Export Data ====================
# compression experiments:
# original .csv: 369mb
# .csv  ->  .csv.gz:   97mb
# .csv  ->  .csv.zip:  98mb
# .csv  ->  .csv.xz:   72mb
# .xz gives really good compression
# .xz will be committed to the repo, .csv will not.

df2.to_csv("btc_usd.csv", index=False)
df2.to_csv("btc_usd.csv.xz", index=False, compression="xz")


# %%
