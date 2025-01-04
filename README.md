# eccv2024-synthesizing-time-varying-BRDFs-via-latent-space

## setup
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


