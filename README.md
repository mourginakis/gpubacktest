# gpubacktest

some people think it's impossible to backtest on a gpu because the problem isn't parallelizable...  
but that's not true.


### Setup
```bash
python3 -m venv venv3.12
source venv3.12/bin/activate
python3 -m pip install -r requirements.txt
```

If you have vscode/cursor you can also run this as a notebook (recommended).
(You need to install the Microsoft Jupyter extension)

I built this so that it should work out of the box. We've downloaded the kaggle dataset into the repository and truncated and compressed it so it's at a size that's friendly for git.


```bash
# you need an nvidia GPU with cuda
nvidia-smi

# you may have to run this:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-4
```

### Notes

- **Bitcoin Historical Data**  
  A comprehensive daily OHLC dataset from Kaggle.  
  🔗 https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data

