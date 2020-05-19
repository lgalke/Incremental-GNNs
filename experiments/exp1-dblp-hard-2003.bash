DATA="dblp-hard"
YEAR=2004
INITIAL_EPOCHS=0
ANNUAL_EPOCHS=200
HISTORY=3
NLAYERS=1
ARGS="--n_layers $NLAYERS --weight_decay 0 --dropout 0.5 --history $HISTORY --rescale_lr 1. --rescale_wd 1."
PRETRAIN_ARGS="--t_start $YEAR --initial_epochs $INITIAL_EPOCHS"
OUTFILE="results-ecml2020-final/exp1-$DATA-$YEAR-$HISTORY.csv"

for SEED in 1 2 3 4 5 6 7 8 9 10; do
  python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --start warm --lr "0.001" --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
  python3 run_experiment.py --seed "$SEED" --model mlp --n_hidden 64 --start cold --lr "0.001" --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
  python3 run_experiment.py --seed "$SEED" --model gs-mean --n_hidden 32 --start warm --lr "0.005" --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
  python3 run_experiment.py --seed "$SEED" --model gs-mean --n_hidden 32 --start cold --lr "0.005" --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
  python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --start warm --lr "0.005" --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
  python3 run_experiment.py --seed "$SEED" --model gat --n_hidden 64 --start cold --lr "0.005" --annual_epochs $ANNUAL_EPOCHS $ARGS $PRETRAIN_ARGS --data "$DATA" --save "$OUTFILE"
done
