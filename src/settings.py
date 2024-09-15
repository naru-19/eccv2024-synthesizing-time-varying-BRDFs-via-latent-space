from pathlib import Path


class Settings:
    dataset_dir = Path("/app/resources/")
    enc_dec_checkpoint = dataset_dir /"model_weights/encdec/600k.pth"