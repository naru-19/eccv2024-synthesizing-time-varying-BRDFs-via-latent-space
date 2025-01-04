# eccv2024-synthesizing-time-varying-BRDFs-via-latent-space

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


