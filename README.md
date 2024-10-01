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
# Citation
Please cite the following work if you make use of this code or data:

```bib
@inproceedings{takenakaViProEnablingControlling2024,
  title = {{{ViPro}}: {{Enabling}} and~{{Controlling Video Prediction}} for~{{Complex Dynamical Scenarios Using Procedural Knowledge}}},
  shorttitle = {{{ViPro}}},
  booktitle = {Neural-{{Symbolic Learning}} and {{Reasoning}}},
  author = {Takenaka, Patrick and Maucher, Johannes and Huber, Marco F.},
  editor = {Besold, Tarek R. and family=Garcez, given=Artur, prefix=d’Avila, useprefix=true and Jimenez-Ruiz, Ernesto and Confalonieri, Roberto and Madhyastha, Pranava and Wagner, Benedikt},
  date = {2024},
  pages = {62--83},
  publisher = {Springer Nature Switzerland},
  location = {Cham},
  doi = {10.1007/978-3-031-71167-1_4},
  isbn = {978-3-031-71167-1},
  langid = {english}
}
```
