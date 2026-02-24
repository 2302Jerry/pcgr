from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

@dataclass
class Sample:
    text: str
    image_path: str
    label: int

class MultimodalCSVDataset(Dataset):
    """CSV with columns: text,image_path,label (label: 0=fake, 1=real)."""

    def __init__(self, csv_path: str, image_root: str = ".", clip_processor=None):
        self.csv_path = str(csv_path)
        self.image_root = Path(image_root)
        self.clip_processor = clip_processor
        self.samples: List[Sample] = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(Sample(
                    text=row["text"],
                    image_path=row["image_path"],
                    label=int(row["label"])
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        img_path = self.image_root / s.image_path
        image = Image.open(img_path).convert("RGB")
        if self.clip_processor is None:
            raise ValueError("clip_processor must be provided.")
        proc = self.clip_processor(text=[s.text], images=[image], return_tensors="pt", padding=True)
        # Squeeze batch dim
        item = {k: v.squeeze(0) for k, v in proc.items()}
        item["label"] = torch.tensor(s.label, dtype=torch.float32)
        item["raw_text"] = s.text
        item["raw_image_path"] = str(img_path)
        return item

def collate_fn(batch: List[Dict]) -> Dict:
    keys = [k for k in batch[0].keys() if k not in ("raw_text", "raw_image_path")]
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["raw_text"] = [b["raw_text"] for b in batch]
    out["raw_image_path"] = [b["raw_image_path"] for b in batch]
    return out
