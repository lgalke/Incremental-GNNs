#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import os.path as osp

import numpy as np
import pandas as pd
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Models
from models import GraphSAGE
from models import GAT
from models import MLP
from models import MostFrequentClass

# # EvolveGCN
# from models.evolvegcn.egcn_o import EGCN
# from models.evolvegcn.models import Classifier
# import models.evolvegcn.utils as egcn_utils

# GCN Sampling
from models.gcn_cv_sc import GCNSampling, GCNInfer, train as train_sampling, copy_params, evaluate as evaluate_sampling, prepare_graph
use_cuda = torch.cuda.is_available()

from datasets import load_data

def appendDFToCSV_void(df, csvFilePath, sep=","):
    """ Safe appending of a pandas df to csv file
    Source: https://stackoverflow.com/questions/17134942/pandas-dataframe-output-end-of-csv
    """
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)


def compute_weights(ts, exponential_decay, initial_quantity=1.0, normalize=True):
    ts = torch.as_tensor(ts)
    delta_t = ts.max() - ts
    values = initial_quantity * torch.exp(- exponential_decay * delta_t)
    if normalize:
        # When normalizing, the initial_quantity is irrelevant
        values = values / values.sum()
    return values


def train(model, optimizer, g, feats, labels, mask=None, epochs=1, weights=None):
    model.train()
    reduction = 'none' if weights is not None else 'mean'
    for epoch in range(epochs):
        logits = model(g, feats)
        if mask is not None:
            loss = F.cross_entropy(logits[mask], labels[mask], reduction=reduction)
        else:
            loss = F.cross_entropy(logits, labels, reduction=reduction)

        if weights is not None:
            loss = (loss * weights).sum()

        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {:d} | Loss: {:.4f}".format(epoch+1, loss.detach().item()))


def evaluate(model, g, feats, labels, mask=None, compute_loss=True):
    model.eval()
    with torch.no_grad():
        logits = model(g, feats)

        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]

        if compute_loss:
            loss = F.cross_entropy(logits, labels).item()
        else:
            loss = None

        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)
        __max_vals, max_indices = torch.max(logits.detach(), 1)
        acc = (max_indices == labels).sum().float() / labels.size(0)

    return acc.item(), loss

def build_model(args, in_feats, n_hidden, n_classes, device, n_layers=1):
    if args.model == 'gcn_cv_sc':
        infer_device = torch.device("cpu")  # for sampling
        train_model = GCNSampling(in_feats,
                                  n_hidden,
                                  n_classes,
                                  2,
                                  F.relu,
                                  args.dropout).to(device)
        infer_model = GCNInfer(in_feats,
                               args.n_hidden,
                               n_classes,
                               2,
                               F.relu)
        model = (train_model, infer_model)
    elif args.model == 'gs-mean':
        model = GraphSAGE(in_feats, n_hidden, n_classes,
                          n_layers, F.relu, args.dropout,
                          'mean').to(device)
    elif args.model == 'mlp':
        model = MLP(in_feats, n_hidden, n_classes,
                    n_layers, F.relu, args.dropout).to(device)
    elif args.model == 'mostfrequent':
        model = MostFrequentClass()
    # elif args.model == 'egcn':
    #     if n_layers != 2:
    #         print("Warning, EGCN doesn't respect n_layers")
    #     egcn_args = egcn_utils.Namespace({'feats_per_node': in_feats,
    #                                       'layer_1_feats': n_hidden,
    #                                       'layer_2_feats': n_classes})
    #     model = EGCN(egcn_args, torch.nn.RReLU(), device=device, skipfeats=False)
    elif args.model == 'gat':
        print("Warning, GAT doesn't respect n_layers")
        heads = [8, args.gat_out_heads]  # Fixed head config
        # Div num_hidden by heads for same capacity
        n_hidden_per_head = int(n_hidden / heads[0])
        assert n_hidden_per_head * heads[0] == n_hidden, f"{n_hidden} not divisible by {heads[0]}"
        model = GAT(1, in_feats, n_hidden_per_head, n_classes,
                    heads, F.elu, 0.6, 0.6, 0.2, False).to(device)
    else:
        raise NotImplementedError("Model not implemented")

    return model

def prepare_data_for_year(graph, features, labels, years, current_year, history, exclude_class=None,
                          device=None):
    print("Preparing data for year", current_year)
    # Prepare subgraph
    subg_nodes = torch.arange(graph.number_of_nodes())[(years <= current_year) & (years >= (current_year - history))]

    subg = graph.subgraph(subg_nodes)
    subg.set_n_initializer(dgl.init.zero_initializer)
    subg_features = features[subg_nodes]
    subg_labels = labels[subg_nodes]
    subg_years = years[subg_nodes]

    # Prepare masks wrt *subgraph*
    train_nid = torch.arange(subg.number_of_nodes())[subg_years < current_year]
    test_nid = torch.arange(subg.number_of_nodes())[subg_years == current_year]

    if exclude_class is not None:
        train_nid = train_nid[subg_labels[train_nid] != exclude_class]
        test_nid = test_nid[subg_labels[test_nid] != exclude_class]

    print("[{}] #Training: {}".format(current_year, train_nid.size(0)))
    print("[{}] #Test    : {}".format(current_year, test_nid.size(0)))
    if device is not None:
        subg_features = subg_features.to(device)
        subg_labels = subg_labels.to(device)
    return subg, subg_features, subg_labels, subg_years, train_nid, test_nid

RESULT_COLS = ['dataset',
               'seed',
               'model',
               'variant',
               'n_params',
               'n_hidden',
               'n_layers',
               'dropout',
               'history',
               'limited_pretraining',
               'initial_epochs',
               'initial_lr',
               'initial_wd',
               'annual_epochs',
               'annual_lr',
               'annual_wd',
               'start',
               'decay',
               'year',
               'epoch',
               'accuracy']

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = '70companies'
    use_sampling = args.model in ['gcn_cv_sc']

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.model == 'mostfrequent':
        # Makes no sense to put things on GPU when using simple most frequent classifier
        device = torch.device("cpu")


    graph, features, labels, years = load_data(args.data_path)

    print("Min year:", years.min())
    print("Max year:", years.max())
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    # graph = dgl.DGLGraph(g_nx, readonly=True)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    years = torch.LongTensor(years)
    n_classes = torch.unique(labels).size(0)

    if not use_sampling:
        features = features.to(device)
        labels = labels.to(device)
        print("Labels", labels.size())

    in_feats = features.shape[1]
    n_layers = args.n_layers
    n_hidden = args.n_hidden

    model = build_model(args, in_feats, n_hidden, n_classes, device, n_layers=args.n_layers)
    if args.model == 'gcn_cv_sc':
        # unzip training and inference models
        model, infer_model = model

    print(model)
    num_params = sum(np.product(p.size()) for p in model.parameters())
    print("#params:", num_params)
    if args.model != 'mostfrequent':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    results_df = pd.DataFrame(columns=RESULT_COLS)
    def attach_score(df, year, epoch, accuracy):
        """ Partial """
        return df.append(
            pd.DataFrame(
                [[osp.basename(osp.normpath(args.data_path)),  # dataset
                  args.seed,
                  args.model,
                  args.variant,
                  num_params,
                  args.n_hidden,
                  args.n_layers,
                  args.dropout,
                  args.history,
                  args.limited_pretraining,
                  args.initial_epochs,
                  args.lr,
                  args.weight_decay,
                  args.annual_epochs,
                  args.lr * args.rescale_lr,
                  args.weight_decay * args.rescale_wd,
                  args.start,
                  args.decay,
                  year,
                  epoch,
                  accuracy]],
                columns=RESULT_COLS),
            ignore_index=True)

    if dataset == 'elliptic':
        exclude_class = 0   # <-- this is the UNK class in the dataset
    else:
        exclude_class = None

    if not args.limited_pretraining and not args.start == 'cold' and args.initial_epochs > 0:
        # With 'limited pretraining' we do the initial epochs on the first wnidow
        # With cold start, no pretraining is needed
        # When initial epochs are 0, no pretraining is needed either
        # For current experiments, we have set initial_epochs = 0
        # Exclusively the static model of experiment 1 uses this pretraining
        subg, subg_features, subg_labels, subg_years, train_nid, test_nid = prepare_data_for_year(graph, features, labels, years, args.pretrain_until, 10000,
                                                                                                  exclude_class=exclude_class, device=device)
        # Use all nodes of initial subgraph for training
        print("Using data until", args.pretrain_until, "for training")
        print("Selecting", subg.number_of_nodes(), "of", graph.number_of_nodes(), "papers for initial training.")


        train_nids = torch.cat([train_nid, test_nid])  # use all nodes in subg for initial pre-training
        if use_sampling:
            prepare_graph(subg, subg_features, n_layers, n_hidden)
            train_sampling(model, optimizer, F.cross_entropy, 1, subg, train_nids, subg_labels, args.initial_epochs,
                  batch_size=args.batch_size, num_workers=args.num_workers)
        elif args.model == 'mostfrequent':
            model.fit(None, subg_labels)
        else:
            print("Subg labels", subg_labels.size())
            train(model, optimizer, subg, subg_features, subg_labels,
                  mask=train_nid,
                  epochs=args.initial_epochs)
            acc, _ = evaluate(model, subg, subg_features, subg_labels, mask=None)
            print(f"** Train Accuracy {acc:.4f} **")


    remaining_years = torch.unique(years[years > args.pretrain_until], sorted=True)

    for t, current_year in enumerate(remaining_years.numpy()):
        torch.cuda.empty_cache() # no memory leaks
        # Get the current subgraph
        subg, subg_features, subg_labels, subg_years, train_nid, test_nid = prepare_data_for_year(graph, features, labels, years, current_year, args.history,
                                                                                                  exclude_class=exclude_class, device=device)

        if args.decay is not None:
            # Use decay factor to weight the loss function, based on time steps t
            if use_sampling:
                raise NotImplementedError("Decay can only be used without sampling")
            weights = compute_weights(years[train_nid], args.decay, normalize=True).to(device)
        else:
            weights = None

        if args.history == 0:
            # No history means no uptraining at all!!!
            # Unused. For the static model (Exp. 1) we give a history frame but do no uptraining instead.
            epochs = 0
        elif args.limited_pretraining and t == 0:
            # Do the pretraining on the first history window
            # with `initial_epochs` instead of `annual_epochs`
            epochs = args.initial_epochs
        else:
            epochs = args.annual_epochs

        # Get a new optimizer with rescaled lr and wd
        if args.start == 'cold':
            del model
            # Build a fresh model for a cold restart
            model = build_model(args, in_feats, n_hidden, n_classes, device, n_layers=args.n_layers)
            if args.model == 'gcn_cv_sc':
                # unzip training and inference models
                model, infer_model = model
        if args.model != 'mostfrequent':
            # Build a fresh optimizer in both cases: warm or cold
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.lr * args.rescale_lr,
                                         weight_decay=args.weight_decay * args.rescale_wd)
        if use_sampling:
            if epochs > 0:
                train_sampling(model, optimizer, F.cross_entropy, 1, subg, train_nid, subg_labels, epochs)
                copy_params(infer_model, model)
            acc = evaluate_sampling(infer_model, subg, test_nid, labels, batch_size=args.test_batch_size, num_workers=args.num_workers)
        elif args.model == 'mostfrequent':
            if epochs > 0:
                # Re-fit only if uptraining is in general allowed!
                model.fit(None, subg_labels[train_nid])
            acc, _ = evaluate(model, subg, subg_features, subg_labels, mask=test_nid, compute_loss=False)
        else:
            if epochs > 0:
                train(model, optimizer, subg, subg_features, subg_labels, mask=train_nid, epochs=epochs,
                      weights=weights)
            acc, _ = evaluate(model, subg, subg_features, subg_labels, mask=test_nid, compute_loss=False)
        print(f"[{current_year} ~ Epoch {epochs}] Test Accuracy: {acc:.4f}")
        results_df = attach_score(results_df, current_year, epochs, acc)
        # input() # debug purposes
        # DROP ALL STUFF COMPUTED FOR CURRENT WINDOW (no memory leaks)
        del subg, subg_features, subg_labels, subg_years, train_nid, test_nid


    if args.save is not None:
        print("Saving final results to", args.save)
        appendDFToCSV_void(results_df, args.save)


DATASET_PATHS = {
    'dblp-easy': 'data/dblp-easy/',
    'dblp-hard': 'data/dblp-hard/',
    'pharmabio': 'data/pharmabio/'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Specify model", default='gs-mean',
                        choices=['mlp','gs-mean','gcn_cv_sc', 'mostfrequent', 'egcn', 'gat'])
    parser.add_argument('--variant', type=str, default='',
                        help="Some comment on the model variant, useful to distinguish within results file")
    parser.add_argument('--dataset', type=str, help="Specify the dataset", choices=list(DATASET_PATHS.keys()),
                        default='pharmabio')
    parser.add_argument('--t_start', type=int,
                        help="The first evaluation time step. Default is 2004 for DBLP-{easy,hard} and 1999 for PharmaBio")

    parser.add_argument('--n_layers', type=int,
                        help="Number of layers/hops", default=2)
    parser.add_argument('--n_hidden', type=int,
                        help="Model dimension", default=64)
    parser.add_argument('--lr', type=float,
                        help="Learning rate", default=0.01)
    parser.add_argument('--weight_decay', type=float,
                        help="Weight decay", default=0.0)
    parser.add_argument('--dropout', type=float,
                        help="Dropout probability", default=0.5)

    parser.add_argument('--initial_epochs', type=int,
                        help="Train this many initial epochs", default=0)
    parser.add_argument('--annual_epochs', type=int,
                        help="Train this many epochs per year", default=200)
    parser.add_argument('--history', type=int,
                        help="How many years of data to keep in history", default=100)

    parser.add_argument('--gat_out_heads',
                        help="How many output heads to use for GATs", default=1, type=int)
    parser.add_argument('--rescale_lr', type=float,
                        help="Rescale factor for learning rate and weight decay after pretraining", default=1.)
    parser.add_argument('--rescale_wd', type=float,
                        help="Rescale factor for learning rate and weight decay after pretraining", default=1.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_neighbors', type=int, default=1,
                        help="How many neighbors for control variate sampling")
    parser.add_argument('--limit', type=int, default=None,
                        help="Debug mode, limit number of papers to load")
    parser.add_argument('--batch_size', type=int, default=1000,
                        help="Number of seed nodes per batch for sampling")
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help="Test batch size (testing is done on cpu)")
    parser.add_argument('--num_workers', type=int, default=8, help="How many threads to use for sampling")
    parser.add_argument('--limited_pretraining', default=False, action="store_true", help="Perform pretraining on the first history window.")
    parser.add_argument('--decay', default=None, type=float, help="Paramater for exponential decay loss smoothing")
    parser.add_argument('--save_intermediate', default=False, action="store_true", help="Save intermediate results per year")
    parser.add_argument('--save', default=None, help="Save results to this file")
    parser.add_argument('--start', default='warm', choices=['cold', 'warm'], help="Cold retrain from scratch or use warm start.")

    ARGS = parser.parse_args()

    if ARGS.save is None:
        print("**************************************************")
        print("*** Warning: results will not be saved         ***")
        print("*** consider providing '--save <RESULTS_FILE>' ***")
        print("**************************************************")

    # Handle dataset argument to get path to data
    try:
        ARGS.data_path = DATASET_PATHS[ARGS.dataset]
    except KeyError:
        print("Dataset key not found, trying to interprete as raw path")
        ARGS.data_path = ARGS.dataset
    print("Using dataset with path:", ARGS.data_path)

    # Handle t_start argument
    if ARGS.t_start is None:
        try:
            ARGS.t_start = {
                    'dblp-easy': 2004,
                    'dblp-hard': 2004,
                    'pharmabio': 1999
                    }[ARGS.dataset]
            print("Using t_start =", ARGS.t_start)
        except KeyError:
            print("No default for dataset '{}'. Please provide '--t_start'.".format(ARGS.dataset))
            exit(1)
    # Backward compatibility:
    # current implementation actually uses 'pretrain_until'
    # as last timestep / year *BEFORE* t_start
    ARGS.pretrain_until = ARGS.t_start - 1

    main(ARGS)
