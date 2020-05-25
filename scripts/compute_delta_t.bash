#!/usr/bin/bash 

python3 compute_historical_connectivity.py --max-hops 1 --save delta_t_with_u/66v-1hop-deltat /data21/lgalke/datasets/dblp-graph-hard
python3 compute_historical_connectivity.py --max-hops 2 --save delta_t_with_u/66v-2hop-deltat /data21/lgalke/datasets/dblp-graph-hard
python3 compute_historical_connectivity.py --max-hops 3 --save delta_t_with_u/66v-3hop-deltat /data21/lgalke/datasets/dblp-graph-hard
python3 compute_historical_connectivity.py --max-hops 1 --save delta_t_with_u/7dc-1hop-deltat /data21/lgalke/datasets/70companies/7dc
python3 compute_historical_connectivity.py --max-hops 2 --save delta_t_with_u/7dc-2hop-deltat /data21/lgalke/datasets/70companies/7dc
python3 compute_historical_connectivity.py --max-hops 3 --save delta_t_with_u/7dc-3hop-deltat /data21/lgalke/datasets/70companies/7dc
python3 compute_historical_connectivity.py --max-hops 1 --save delta_t_with_u/12v-1hop-deltat /data21/lgalke/datasets/dblp-graph
python3 compute_historical_connectivity.py --max-hops 2 --save delta_t_with_u/12v-2hop-deltat /data21/lgalke/datasets/dblp-graph
python3 compute_historical_connectivity.py --max-hops 3 --save delta_t_with_u/12v-3hop-deltat /data21/lgalke/datasets/dblp-graph
# python3 compute_historical_connectivity.py --max-hops 1 --save delta_t_with_u/elliptic-1hop-deltat /data21/lgalke/datasets/elliptic_bitcoin_dataset
# python3 compute_historical_connectivity.py --max-hops 2 --save delta_t_with_u/elliptic-2hop-deltat /data21/lgalke/datasets/elliptic_bitcoin_dataset
# python3 compute_historical_connectivity.py --max-hops 3 --save delta_t_with_u/elliptic-3hop-deltat /data21/lgalke/datasets/elliptic_bitcoin_dataset



# usage: analyze_historic_connectivity.py [-h] [--max-hops MAX_HOPS]
#                                         [--save SAVE]
#                                         data_path

# positional arguments:
#   data_path

# optional arguments:
#   -h, --help           show this help message and exit
#   --max-hops MAX_HOPS
#   --save SAVE
