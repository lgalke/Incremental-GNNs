DATA="dblp-easy"
YEAR=2004
INITIAL_EPOCHS=0
ANNUAL_EPOCHS=200
HISTORY=3
NLAYERS=1
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --history $HISTORY --rescale_lr 1. --rescale_wd 1."
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITIAL_EPOCHS"
OUTFILE="results-ecml2020-final/exp1-hpopt-ours.csv"

for SEED in 101 102 103; do
  for LR in "0.0001" "0.0005" "0.001" "0.005" "0.01" "0.05" "0.01"; do
    python3 run_experiment.py --seed "$SEED" --variant "augmentation" --model ours --n_hidden 32 --start warm --lr $LR --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
    python3 run_experiment.py --seed "$SEED" --variant "augmentation" --model ours --n_hidden 32 --start cold --lr $LR --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
  done
done
