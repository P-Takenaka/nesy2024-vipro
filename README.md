# Preparations
1. Download the datasets from https://drive.google.com/drive/folders/1LnPt6XwJtsOI0J3tadkhZZIRkWNUdhUZ?usp=sharing
2. Unpack the datasets
3. Set the environment variables in .env
4. Create and activate conda environment
```
conda create --file environment.yml
conda activate vipro
```

Start training with a config within the repo, e.g.
```
python3 train.py --config=src/configs/orbits/learned_params.py
```
