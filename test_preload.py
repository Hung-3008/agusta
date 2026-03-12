import time
import yaml
from torch.utils.data import DataLoader
from src.data.algonauts_dataset import AlgonautsDataset, resolve_paths

with open('src/configs/brainflow.yaml') as f:
    cfg = yaml.safe_load(f)

cfg = resolve_paths(cfg, '.')

print("Building train dataset and preloading to RAM...")
ds = AlgonautsDataset(cfg, split="train", max_cache_gb=10.0)

t0 = time.time()
ds.preload_to_ram()
print(f"Preload finished in {time.time()-t0:.2f}s")
