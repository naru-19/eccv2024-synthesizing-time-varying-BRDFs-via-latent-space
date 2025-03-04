import argparse
from pathlib import Path

import numpy as np
import torch
from brdf_enc_dec.model import BRDFNPs, ModelConfig
from brdf_enc_dec.torrance_sparrow_dataset import TorspaBRDFNPsDataset
from staf import STAF
from TorranceSparrow import TorranceSparrowParams, torspa
from tqdm import tqdm

from settings import get_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvBTF", type=int, default=45)
    parser.add_argument("--poly", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--out_dir", type=Path, default="./output")
    args = parser.parse_args()
    device = f"cuda:{args.gpu}" if args.gpu != "-1" else "cpu"
    settings = get_settings()

    # load brdf encoder/decoder
    brdfnps = BRDFNPs(
        device=device,
        checkpoint=settings.enc_dec_checkpoint,
        onlyEncoder=True,
        cfg=ModelConfig(),
    )
    brdfnps.to_eval()

    # load tvBRDF data
    staf = STAF.parse_files(
        settings.dataset_dir / f"tvBTF{args.tvBTF}/poly-{args.poly}"
    )

    # compress into latent space
    latents_all = []
    out_root_dir = args.out_dir
    for frame, t in enumerate(np.linspace(0, 1, settings.frames)):
        print(f"frame {frame+1}/{settings.frames}")
        out_dir = out_root_dir / f"t-{t:.2f}"
        workspace_dir = out_dir / "batch_output"
        workspace_dir.mkdir(exist_ok=True, parents=True)

        # build dataset
        tsp = staf.get_tsp(t=t, device=device)
        dataset = TorspaBRDFNPsDataset(
            params=tsp, batch_size=args.batch_size, device=device
        )
        results = []
        with torch.no_grad():
            for i in tqdm(range(dataset.num_baches)):
                batch_input = dataset.get_batch(i).reshape(-1, 16200, 7)
                latent = brdfnps.encode(batch_input)
                results.append(latent.cpu().numpy())
                np.save(workspace_dir / f"batch_{i}.npy", latent.cpu().numpy())
            results = np.concatenate(results, axis=0)
            np.save(out_dir / f"f{frame}_latents_all.npy", results)
            latents_all.append(
                results[..., : ModelConfig().latent_dim].reshape(-1, 1, 10)
            )


if __name__ == "__main__":
    main()
