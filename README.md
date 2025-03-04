# Synthesizing time-varying BRDFs via latent space (ECCV2024)
This repository is the official implementation of the ECCV 2024 paper "Synthesizing time-varying BRDFs via latent space"


## Requirements
We used the following environement for the experiments:
- Python 3.8

Other dependencies can be installed using pip as follows:
```
pip install -r requirements.txt
cd libraries && pip install -r requirements.txt
```
## setup

download latens files from [google drive](https://drive.google.com/file/d/1XATLeXinE3jUW_T2dGBlS5Coc5iAz-FZ/view?usp=sharing)
```
make run
cd libraries/
pip install -r requirements.txt
```

### train/run script
**compress tvbrdf into latent**
```
python3 compress_tvbrdf.py
```

**train ntm**
```
python3 train_ntm.py --case rust
```


## Citation
```
@inproceedings{narumoto2024synthesizing,
  title={Synthesizing Time-Varying BRDFs via Latent Space},
  author={Narumoto, Takuto and Santo, Hiroaki and Okura, Fumio},
  booktitle={European Conference on Computer Vision},
  pages={109--124},
  year={2024},
  organization={Springer}
}
```