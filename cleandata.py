#%% ==================== Imports ====================
import pandas as pd


#%% ==================== Load Data ====================
# downloaded data from kaggle (~300mb)
# https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
df = pd.read_csv("btcusd_1-min_data.csv")
df.info(memory_usage="deep")


#%% ==================== Clean Data ====================
# keep only timestamp, close
# reduces size from 369mb to 107mb
df1 = df[["Timestamp", "Close"]]
df1.info(memory_usage="deep")


#%% ==================== Export Data ====================
# compression experiments:
# original .csv: 369mb
# .csv  ->  .csv.gz:   97mb
# .csv  ->  .csv.zip:  98mb
# .csv  ->  .csv.xz:   72mb
# .xz gives really good compression
# .xz will be committed to the repo, .csv will not.

df1.to_csv("btc_usd.csv", index=False)
df1.to_csv("btc_usd.csv.xz", index=False, compression="xz")


# %%
