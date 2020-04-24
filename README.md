# Incremental Training of Graph Neural Networks

Paper: under review at ECML 2020


## Installation

1. Install [pytorch](https://pytorch.org/get-started/locally/) as suited to your
   OS / python package manager / CUDA version
1. Install [dgl](https://www.dgl.ai/pages/start.html) as suited to your
   OS / python package manager / CUDA version
1. Install other requirements via `pip install -r requirements.txt` within your
   copy of this repository. This will include mainly `pandas` and `seaborn`.

## Get the datasets

The three datasets of our paper are available [on zenodo](https://zenodo.org/deposit/3764770).
Download the zip files and put them into the `data` subdirectory:

- `data/dblp-easy`
- `data/dblp-hard`
- `data/pharmabio`

## Run an experiment

```
python3 run_experiment 
```

## Visualize results

```
python3
```

## File Descriptions

| File                   | Description                                      |
| -                      | -                                                |
| analysis               | scripts to perform analyses                      |
| datasets.py            | dataset loading                                  |
| experiments            | scripts to reproduce experiments                 |
| models                 | GNN implementations                              |
| README.md              | this file                                        |
| requirements.txt       | dependencies                                     |
| run_experiment.py      | main entry point for running a single experiment |
| tabularize_ecml2020.py | reproduce results table from ECML2020 submission |
| tabularize.py          | aggregate results into table                     |
| visualize.py           | visualize results                                |
