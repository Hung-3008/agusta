import time
import yaml
from torch.utils.data import DataLoader
from src.data.algonauts_dataset import AlgonautsDataset, resolve_paths

with open('src/configs/brainflow.yaml') as f:
    cfg = yaml.safe_load(f)

cfg = resolve_paths(cfg, '.')

print("Building train dataset...")
ds = AlgonautsDataset(cfg, split="train", max_cache_gb=10.0)

print("Starting to load 10 items sequentially...")
start = time.time()
for i in range(10):
    t0 = time.time()
    ds[i]
    print(f"Item {i} loaded in {time.time()-t0:.4f}s")
print(f"Total for 10 sequential items: {time.time()-start:.4f}s")

print("Starting to load 10 random items...")
import random
indices = random.sample(range(len(ds)), 10)
start = time.time()
for i, idx in enumerate(indices):
    t0 = time.time()
    ds[idx]
    print(f"Random Item {idx} loaded in {time.time()-t0:.4f}s")
print(f"Total for 10 random items: {time.time()-start:.4f}s")

