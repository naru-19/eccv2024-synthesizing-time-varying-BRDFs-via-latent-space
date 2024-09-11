import sys
from typing import List, Optional

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from pydantic_settings import BaseSettings
from torch.optim.lr_scheduler import LambdaLR


class BaseModelConfig(BaseSettings):
    rgb_latent: int
    brdf_latent: int
    num_log1p: int
    latent_dim: int
    learning_rate: float
    seed: int


class ModelConfig:
    rgb_latent = 3
    brdf_latent = 7
    num_log1p = 4
    latent_dim = rgb_latent + brdf_latent
    learning_rate = 1e-4
    seed = 2023


class residual(nn.Module):
    def __init__(self, module):
        super(residual, self).__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class FCLayerNormReLU(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        torch.nn.init.zeros_(self.fc.bias)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.fc(input)))


# class BRDFNPs:
class FCReLU(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(input))


class Aggregator(nn.Module):
    def __init__(self, latent_dim: int = 7, in_features: int = 64) -> None:
        """
        Aggregator a
        input: s 64dim
        return: zk (myu: latent_dim - 1,sigma: 1dim)
        """
        super().__init__()
        self.aggregate = nn.Sequential(
            FCReLU(in_features=in_features, out_features=128),
            FCReLU(in_features=128, out_features=128),
        )
        self.linear_for_mean = nn.Linear(in_features=128, out_features=latent_dim)
        self.linear_for_var = nn.Linear(in_features=128, out_features=latent_dim)
        self.latent_dim = latent_dim

    def mean_pooling(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: N/M x 64 dim
        return 64dim vector
        """
        return torch.nanmean(s, dim=-2)

    def forward(self, input: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is not None:
            input = input + mask
        zk_middle = self.aggregate(self.mean_pooling(input))
        mu = self.linear_for_mean(zk_middle)
        sigma = self.linear_for_var(zk_middle)
        return mu, sigma


class FCLayerNormReLU(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        torch.nn.init.zeros_(self.fc.bias)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.fc(input)))


class BRDFEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_first = nn.MultiheadAttention(
            embed_dim=64, num_heads=8, batch_first=True
        )
        self.attn_second = nn.MultiheadAttention(
            embed_dim=64, num_heads=8, batch_first=True
        )

        self.upsampler = nn.Sequential(
            FCLayerNormReLU(in_features=5, out_features=64),
            nn.Dropout(0.2),
            residual(
                nn.Sequential(
                    FCLayerNormReLU(in_features=64, out_features=64),
                    FCReLU(in_features=64, out_features=64),
                )
            ),
            nn.Dropout(0.2),
            FCLayerNormReLU(in_features=64, out_features=64),
        )

        self.mlp = nn.Sequential(
            FCReLU(in_features=64, out_features=400),
            FCReLU(in_features=400, out_features=400),
            FCReLU(in_features=400, out_features=64),
        )

    def forward(
        self, input: torch.Tensor, cls_token: torch.tensor, mask=None
    ) -> torch.Tensor:
        """
        input: batch size x sample x 7
        cls_token: 1 x 64
        return: batch size x sample x 64
        """
        if mask is not None:
            bs, _, _ = input.shape
            nan_mask = torch.isnan(mask[..., :7])
            input = input[~nan_mask]
            input = input.reshape(bs, -1, 7)

        # attention入力前処理
        r = torch.cat([input[..., :4], input[..., 4:5]], dim=-1)
        g = torch.cat([input[..., :4], input[..., 5:6]], dim=-1)
        b = torch.cat([input[..., :4], input[..., 6:7]], dim=-1)

        r = self.upsampler(r).reshape(-1, 1, 64)
        g = self.upsampler(g).reshape(-1, 1, 64)
        b = self.upsampler(b).reshape(-1, 1, 64)
        # 一度目のattention
        attn_input = torch.cat([r, g, b], dim=-2)  # (batch x sample) x 3 x 64
        attn_output, _ = self.attn_first(attn_input, attn_input, attn_input)

        cls_token = torch.tile(
            cls_token.reshape(1, 1, -1), (attn_output.shape[0], 1, 1)
        )
        # 二度目のattention
        attn_input = torch.cat([attn_output, cls_token], dim=-2)
        attn_output, _ = self.attn_second(attn_input, attn_input, attn_input)

        attn_feature = attn_output[:, -1, :].reshape(*input.shape[:2], -1)

        # MLP
        output = self.mlp(attn_feature)
        return output


class Encoder(nn.Module):
    def __init__(self, cfg: BaseModelConfig):
        super().__init__()
        self.rgb_agg = Aggregator(latent_dim=cfg.rgb_latent, in_features=64)
        # 色に依存しない特徴抽出
        self.brdf_encoder = BRDFEncoder()
        self.brdf_agg = Aggregator(latent_dim=cfg.brdf_latent, in_features=64)

    def rgb_encoder(self, input: torch.tensor, mask=None) -> torch.Tensor:
        if mask is not None:
            input = input + mask[..., :7]
        return torch.nanmedian(input[:, :, -3:], dim=-2).values

    def forward(
        self, input: torch.Tensor, cls_token: torch.Tensor, mask=None
    ) -> torch.Tensor:
        rgb_mu = self.rgb_encoder(input, mask)
        assert rgb_mu.shape[-1] == 3
        brdf_mu, brdf_sigma = self.brdf_agg(self.brdf_encoder(input, cls_token, mask))
        latent = torch.cat([rgb_mu, brdf_mu, brdf_sigma], dim=-1)
        return latent


class Decoder(nn.Module):
    def __init__(self, in_features: int = 11) -> None:
        """
        decoder g
        input: zk+x
        return: y_pred
        """
        super().__init__()
        self.decode = nn.Sequential(
            FCReLU(in_features=in_features, out_features=400),
            residual(
                nn.Sequential(
                    residual(
                        nn.Sequential(
                            FCReLU(in_features=400, out_features=400),
                            nn.Dropout(0.2),
                            FCReLU(in_features=400, out_features=400),
                        )
                    ),
                    FCReLU(in_features=400, out_features=400),
                    residual(
                        nn.Sequential(
                            FCReLU(in_features=400, out_features=400),
                            nn.Dropout(0.2),
                            FCReLU(in_features=400, out_features=400),
                        )
                    ),
                )
            ),
            FCReLU(in_features=400, out_features=3),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.decode(input)


class BRDFNPs:
    def __init__(
        self,
        device,
        cfg: BaseModelConfig = ModelConfig(),
        checkpoint: str = None,
        onlyDecoder=False,
        onlyEncoder=False,
    ):
        print("model path:", checkpoint)
        self.device = device
        self.encoder = Encoder(cfg=cfg).to(device)
        self.decoder = Decoder(cfg.latent_dim + 4).to(device)

        print("Set up models...")
        if onlyDecoder:
            print("device:", device)
            self.decoder = Decoder(cfg.latent_dim + 4).to(device)
            print("loaded")
            _ = self.only_decoder_mode(checkpoint)

            return
        if onlyEncoder:
            print("device:", device)
            self.encoder = Encoder().to(device)
            print("loaded")
            _ = self.only_encoder_mode(checkpoint)
            self.encoder = self.encoder.to(device)
            self.cls_token = self.cls_token.to(device)
            return

        self.optimizer_e = torch.optim.Adam(
            self.encoder.parameters(), lr=cfg.learning_rate
        )
        self.optimizer_d = torch.optim.Adam(
            self.decoder.parameters(), lr=cfg.learning_rate
        )
        torch.manual_seed(cfg.seed)
        self.cls_token = torch.randn(1, 64).to(device)
        self.cls_token.requires_grad = True

        self.optimizer_token = torch.optim.Adam(
            [{"params": self.cls_token, "lr": cfg.learning_rate}]
        )
        print("model setup done")
        itr, _ = self.load(checkpoint)
        print(
            "model loaded from ",
            checkpoint,
        )

    def save(self, fpath, itr, seed):
        torch.save(
            {
                "seed": seed,
                "itr": itr,
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "optimizer_e": self.optimizer_e.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
                "optimizer_token": self.optimizer_token.state_dict(),
                "cls_token": self.cls_token,
            },
            fpath,
        )

    def load(self, fpath):
        self.encoder.load_state_dict(
            torch.load(fpath, map_location=self.device)["encoder"]
        )
        self.encoder = self.encoder.to(self.device)
        self.decoder.load_state_dict(
            torch.load(fpath, map_location=self.device)["decoder"]
        )
        self.decoder = self.decoder.to(self.device)
        self.optimizer_e.load_state_dict(
            torch.load(fpath, map_location=self.device)["optimizer_e"]
        )
        self.optimizer_d.load_state_dict(
            torch.load(fpath, map_location=self.device)["optimizer_d"]
        )
        self.cls_token = torch.load(fpath, map_location=self.device)["cls_token"]
        return torch.load(fpath)["itr"], torch.load(fpath)["seed"]

    def backward_step(self, loss):
        self.optimizer_e.zero_grad()
        self.optimizer_d.zero_grad()
        self.optimizer_token.zero_grad()
        loss.backward()
        self.optimizer_e.step()
        self.optimizer_d.step()
        self.optimizer_token.step()

    def encode(self, input, mask=None):
        return self.encoder(input, self.cls_token, mask)

    def decode(self, input):
        return self.decoder(input)

    def to_train(self):
        self.encoder.train()
        self.decoder.train()

    def to_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def only_decoder_mode(self, fpath):
        self.decoder.load_state_dict(torch.load(fpath, map_location="cpu")["decoder"])
        self.decoder = self.decoder.to(self.device)
        return

    def only_encoder_mode(self, fpath):
        self.encoder.load_state_dict(torch.load(fpath, map_location="cpu")["encoder"])
        self.encoder = self.encoder.to(self.device)
        self.cls_token = torch.load(fpath, map_location=self.device)["cls_token"]
        return
