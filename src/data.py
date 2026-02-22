from __future__ import annotations

from PIL import Image
import torch


class MultiLabelImageList:
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        if isinstance(item, dict):
            path = item["path"]
            labels = item["labels"]
        else:
            path, labels = item
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(labels, dtype=torch.float32)
        return image, target
