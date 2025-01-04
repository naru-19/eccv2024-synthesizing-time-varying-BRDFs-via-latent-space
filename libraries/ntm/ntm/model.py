import math
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR


def lrfunc_for_lrsearch(epoch):
    return 2 ** (epoch)


def get_scheduler_for_lrsearch(optimizer):
    return LambdaLR(optimizer, lr_lambda=lrfunc_for_lrsearch)


def lrfunc(epoch):
    return 1


def get_scheduler(optimizer):
    return LambdaLR(optimizer, lr_lambda=lrfunc)


class _NTM(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        end_frame: int = 36,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.end_frame = end_frame
        # rnn
        self.lstm_brdf = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=0.2,
        )
        self.lstm_rgb = nn.LSTM(
            input_size=16,
            hidden_size=16,
            batch_first=True,
            num_layers=num_layers,
            dropout=0.2,
        )

        # mlp
        self.mlp_brdf_first = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
        )
        self.mlp_rgb_first = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )

        self.mlp_brdf_last = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7),
        )

        self.mlp_rgb_last = nn.Sequential(
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 1, 10)
        return:
            x_1_end: (batch_size, 35, 10)
        """
        x_1_end = []

        rgb = x[..., :3]
        brdf = x[..., 3:]

        h_brdf = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_brdf = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        h_rgb = torch.zeros(self.num_layers, x.size(0), 16).to(x.device)
        c_rgb = torch.zeros(self.num_layers, x.size(0), 16).to(x.device)
        for _ in range(self.end_frame - 1):
            brdf_input = self.mlp_brdf_first(brdf)
            rgb_input = self.mlp_rgb_first(rgb)
            rgb_out, (h_rgb, c_rgb) = self.lstm_rgb(rgb_input, (h_rgb, c_rgb))
            rgb = self.mlp_rgb_last(rgb_out) + rgb

            brdf_out, (h_brdf, c_brdf) = self.lstm_brdf(brdf_input, (h_brdf, c_brdf))
            brdf = self.mlp_brdf_last(brdf_out) + brdf

            x_1_end.append(torch.cat([rgb, brdf], dim=-1))
        return torch.cat(x_1_end, dim=-2)


class NTM:
    def __init__(
        self,
        device: str = "cuda:0",
        lr: float = 0.0001,
        in_features: int = 7,
        out_features: int = 7,
        end_frame: int = 36,
    ):
        # config
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.end_frame = end_frame

        self.lr = lr
        # model setup
        self.model = _NTM()
        self.model.to(self.device)
        # setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )
        self.scheduler = get_scheduler(self.optimizer)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def backward_loss(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()
        else:
            pass

    def predict(self, x):
        return self.model(x.to(self.device))

    def save(self, fpath: Union[str, Path], epoch: int) -> None:
        if self.scheduler is not None:
            torch.save(
                {
                    "epoch": epoch,
                    "net": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                fpath,
            )
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "net": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                fpath,
            )

    def load(self, fpath: Union[str, Path]) -> int:
        state_dict = torch.load(fpath, map_location=self.device)
        self.model.load_state_dict(state_dict["net"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        return state_dict["epoch"]
