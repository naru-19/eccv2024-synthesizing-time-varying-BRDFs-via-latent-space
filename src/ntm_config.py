from pathlib import Path

from pydantic_settings import BaseSettings

N_SAMPLE = 1


class BaseConfig(BaseSettings):
    latent_dim: int = 10
    project: str = "ntm-real"
    batch_size: int = 1024
    epochs: int = 1000

    sample: int = 90000
    seed: int = 2023
    split_rate: float = 0.75
    lr: float = 0.0005
    end_frame: int = 36
    data_length: int = 10000
    size: int = 300
    lambda1: int = 0

    latent_dir: Path = Path(__file__).absolute().parent.parent / "resources/latents"


class ConfigRust(BaseConfig):
    data_path: Path = BaseConfig().latent_dir / "tv45.npy"

    suffix: str = "rust"


class ConfigBurn(BaseConfig):
    data_path: Path = BaseConfig().latent_dir / "tv09.npy"
    suffix: str = "burn"


class ConfigPat(BaseConfig):
    data_path: Path = BaseConfig().latent_dir / "tv44.npy"
    suffix: str = "pat"


def get_config(case: str) -> BaseConfig:
    return eval("Config" + case.capitalize() + "()")
