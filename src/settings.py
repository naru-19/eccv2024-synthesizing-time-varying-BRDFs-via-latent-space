import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    dataset_dir: Path = Path(__file__).absolute().parent.parent / "resources"
    enc_dec_checkpoint: Path = dataset_dir / "model_weights/encdec/600k.pth"

    frames: int = 36

    pot_norm_path: Path = dataset_dir / "env" / "potnorm.npy"
    env_map_path: Path = dataset_dir / "env" / "env.pkl"


def get_settings():
    return Settings()


def main():
    assert Settings().dataset_dir.exists()
    assert Settings().enc_dec_checkpoint.exists()


if __name__ == "__main__":
    main()
