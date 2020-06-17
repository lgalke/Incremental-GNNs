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

The three datasets of our paper are available [on zenodo](https://zenodo.org/record/3764770).
Download the zip files and extract them into the `data` subdirectory, such that the structure looks exactly like this:

- `data/dblp-easy`
- `data/dblp-hard`
- `data/pharmabio`

## Example call to run an experiment

The following exemplary command will run an experiment with a GraphSAGE model (1 hidden layer with 32 hidden units) on the `dblp-easy` dataset starting evaluation at year 2003 while using 200 annual epochs.

```
python3 run_experiment.py --seed 42 --model gs-mean --n_hidden 32 --start cold --lr "0.005" --history 3 --n_layers 1 --weight_decay 0 --dropout 0.5 --rescale_lr 1.0 --initial_epochs 0 --annual_epochs 200 --dataset "dblp-easy" --t_start 2003  --save "results.csv"                       
```

The results.csv file can be reused for multiple runs (e.g. with different seeds, different models, different datasets), the script appends new results to the file.
Consult `python3 run_experiment.py -h` for more information.


## Visualize results

You can visualize with the `visualize.py` script:

```
python3 visualize.py --style "window size %RF" --hue model --col dataset --row start --nosharey --save plot.png results.csv
```

where results.csv is the file where you previously aggregated results. You can also provide multiple results files, then they will be concatenated before plotting.

## Full reproduction of the paper's experiments

In the `experiments/` directory, you find bash scripts to re-run all of our experiments (this may take a while).

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
