from typing import Union

import numpy as np
import torch
from ntm.model import NTM
from tqdm import tqdm


class TransferDataset(torch.utils.data.Dataset):
    def __init__(self, src: torch.Tensor, device=None) -> None:
        assert len(src.shape) == 2 or (
            src.shape[1] == 1 and len(src.shape) == 3
        ), f"src shape must be (N,10) or (N,1,10), but got {src.shape}"
        if not isinstance(src, torch.Tensor):
            src = torch.tensor(src)
        if device is not None:
            src = src.to(device)
        self.src = src.to(torch.float32)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx].reshape(1, -1)


class Transfer:
    def __init__(self, ntm: NTM):
        self.ntm = ntm
        self.device = ntm.device
        self.ntm.eval()
        print("eval mode")

    def run(self, src: Union[torch.Tensor, np.ndarray], batch_size=512):
        if isinstance(src, np.ndarray):
            src = torch.tensor(src)
        src = src.to(self.device).to(torch.float32)
        dataset = TransferDataset(src, device=self.device)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        result = []
        with torch.no_grad():
            for batch_input in tqdm(loader):
                dst = self.ntm.predict(batch_input)
                result.append(dst)
        result = torch.cat(result, dim=0)
        result = torch.cat([src.reshape(src.shape[0], 1, -1), result], dim=1)
        return result
