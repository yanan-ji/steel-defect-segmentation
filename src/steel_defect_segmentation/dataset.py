"""OpenCV/PyTorch image dataset and synchronized patch transform."""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Mapping, Sequence
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .labels import decode_multilabel

class SteelDefectDataset(Dataset):
    def __init__(self, image_dir: str | Path, image_ids: Sequence[str], labels: Mapping[str, Mapping[int, object]], transform: Callable | None = None):
        self.image_dir, self.image_ids, self.labels, self.transform = Path(image_dir), list(image_ids), labels, transform
    def __len__(self): return len(self.image_ids)
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(str(self.image_dir / image_id), cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(self.image_dir / image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = decode_multilabel(self.labels.get(image_id, {}), *image.shape[:2])
        if self.transform is not None: image, mask = self.transform(image, mask)
        return {"image": torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float().div(255), "mask": torch.from_numpy(np.ascontiguousarray(mask)).float(), "image_id": image_id}

class RandomCropOrResize:
    """Synchronized training patch operation; masks always use nearest interpolation."""
    def __init__(self, height=256, width=512, seed=None): self.height, self.width, self.rng = height, width, np.random.default_rng(seed)
    def __call__(self, image, mask):
        h, w = image.shape[:2]
        if h < self.height or w < self.width:
            new_w, new_h = max(w, self.width), max(h, self.height)
            image = cv2.resize(image, (new_w,new_h), interpolation=cv2.INTER_LINEAR)
            mask = np.stack([cv2.resize(channel, (new_w,new_h), interpolation=cv2.INTER_NEAREST) for channel in mask])
            h, w = new_h, new_w
        top, left = self.rng.integers(0,h-self.height+1), self.rng.integers(0,w-self.width+1)
        return image[top:top+self.height,left:left+self.width], mask[:,top:top+self.height,left:left+self.width]
