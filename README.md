# gpubacktest

A lot of people told me this was impossible to do, so building it was 
extremely vindicating. Now it's open source.

```bash
# ------ Output (Ryzen 5 3600, RTX 3080Ti):
# CPU Backtest (naive):    117.18413124  seconds per run
# CPU multi-core Backtest: 0.05004053    seconds per run
# GPU Backtest:            0.01131282    seconds per run
# Max speedup: 10358.529x
```

Yes, it's literally 10,000x faster than the naive CPU version.


### Setup
```bash
python3.12 -m venv venv3.12
source venv3.12/bin/activate
python3 -m pip install -r requirements.txt
```

If you have vscode/cursor you can also run this as a notebook (recommended).
(You need to install the Microsoft Jupyter extension)

I built this so that it should work out of the box. We've downloaded the kaggle dataset into the repository and truncated and compressed it so it's at a size that's friendly for git. It gets extracted on the fly when you run the notebook.


```bash
# you need an nvidia GPU with cuda
nvidia-smi

# you may have to run this:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-4
```


If you're at all like me, you'll want to be able to play with this from your laptop while your heavy desktop does all the GPU stuff.

```bash
# start ssh on your host
sudo apt install openssh-server
sudo systemctl restart ssh
```
On local, install the Remote-SSH extension pack in 
vscode/cursor. Connect and you should be able to start up a jupyter IPython kernel just like normal. This is a really amazing workflow.



### Notes

- **Bitcoin Historical Data**  
  A comprehensive daily OHLC dataset from Kaggle.  
  🔗 https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data

