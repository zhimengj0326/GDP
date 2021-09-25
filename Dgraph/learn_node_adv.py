"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import sys
import os
import random
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
# from utils import EarlyStopMonitor

from module import TGAN
from graph import NeighborFinder
from kde import kde_fair


class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)

class Adversary(nn.Module):

    def __init__(self, n_input, n_sensitive, n_hidden=32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))

def pretrain_adversary(idx_list, train_src_l, train_ts_l, n_label, Sn_feat, \
                    clf, adv, tgan, adv_optimizer, adv_criterion, epochs):

    num_test_instance = len(train_src_l)
    num_test_batch = math.ceil(num_test_instance / BATCH_SIZE)
    for epoch in tqdm(range(epochs)):
        np.random.shuffle(idx_list)
        tgan = tgan.eval()
        clf= clf.eval()
        adv = adv.train()
        #num_batch
        for k in range(num_test_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = train_src_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            # label_l_cut = train_label_l[s_idx:e_idx]
            label_l_cut = n_label[src_l_cut]
            Sn_feat_cut = Sn_feat[src_l_cut]
            
            adv_optimizer.zero_grad()
            with torch.no_grad():
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
            
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            src_Sn_feat = torch.from_numpy(Sn_feat_cut).float().to(device)

            lr_prob = clf(src_embed).sigmoid()
            lr_prob = torch.unsqueeze(lr_prob, 1)
            p_z = adv(lr_prob).flatten()

            adv_loss = adv_criterion(p_z, src_Sn_feat) * args.hyper_pent
            adv_loss.backward()
            adv_optimizer.step()

    return adv

def pretrain_classifier(idx_list, train_src_l, train_ts_l, n_label, Sn_feat, \
                        clf, tgan, clf_optimizer, clf_criterion, epochs):
    num_test_instance = len(train_src_l)
    num_test_batch = math.ceil(num_test_instance / BATCH_SIZE)
    for epoch in tqdm(range(epochs)):
        np.random.shuffle(idx_list)
        tgan = tgan.eval()
        clf = clf.train()
        #num_batch
        for k in range(num_test_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = train_src_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            # label_l_cut = train_label_l[s_idx:e_idx]
            label_l_cut = n_label[src_l_cut]
            Sn_feat_cut = Sn_feat[src_l_cut]
            
            clf_optimizer.zero_grad()
            with torch.no_grad():
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
            
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            src_Sn_feat = torch.from_numpy(Sn_feat_cut).float().to(device)

            lr_prob = clf(src_embed).sigmoid()
            lr_loss = clf_criterion(lr_prob, src_label)
            lr_loss.backward()
            clf_optimizer.step()

    return clf


random.seed(222)
np.random.seed(222)
torch.manual_seed(222)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=30, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=50, help='number of neighbors to sample')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=None, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=None, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

parser.add_argument('--running_times', type=int, default=5, help='number of running times')
parser.add_argument('--day_times', type=int, default=1, help='number of recording day in the dataset')
parser.add_argument('--features_type', type=str, default='s', help='type of sensitive attributes (h or s)')
parser.add_argument('--clf', type=str, choices=['clf', 'reg'], default='clf', help='nodel classificaltion/regression')
parser.add_argument('--sens_bn', type=str, choices=['yes', 'no'], default='no', help='sensitive attributes binary')
parser.add_argument('--hyper_pent', type=float, default=1.0, help='Hyperparmeters for penalty')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
RUNNING_TIME = args.running_times
time_duration = args.day_times
features_type = args.features_type
hyper_pent = args.hyper_pent

# code_root_path = './code/'
code_root_path = './'
data_root_path = '/data/zhimengj/dataset/Harris/'

### Load data and train val test split
if DATA=='harris':
    g_df = pd.read_csv(data_root_path + '{}_edge_{}.csv'.format(DATA, time_duration))
    e_feat = np.load(data_root_path + '{}_edge_{}.npy'.format(DATA,time_duration))
    # print(f'e_feat={e_feat.shape}')
    n_feat = np.load(data_root_path + '{}_node.npy'.format(DATA))
    if args.clf=='clf':
        n_label = np.load(data_root_path + '{}_Ynode.npy'.format(DATA))
    else:
        n_label = np.load(data_root_path + '{}_Ynode2.npy'.format(DATA))


if DATA=='harris':
    val_time, test_time = 0.60 * time_duration * 6, 0.8 * time_duration * 6
else:
    val_time, test_time = list(np.quantile(g_df.ts, [0.60, 0.80]))
# print(f'val_time={val_time}')
# print(f'test_time={test_time}')

if DATA=='harris':
    Sn_feat = pickle.load( open(data_root_path + '{}_Snode.p'.format(DATA), "rb") )
    SHn_feat = (Sn_feat > np.mean(Sn_feat)).astype(int)

# val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
# label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))

def eval_epoch_clf(src_l, dst_l, ts_l, n_label, batch_size, lr_model, tgan, num_layer=NODE_LAYER):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    pred_prob = np.zeros(len(src_l))
    loss = 0
    num_instance = len(src_l)
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
        lr_model.eval()
        tgan.eval()
        for k in range(num_batch):          
            s_idx = k * batch_size
            e_idx = min(num_instance - 1, s_idx + batch_size)
            src_l_cut = src_l[s_idx:e_idx]
            # dst_l_cut = dst_l[s_idx:e_idx]
            ts_l_cut = ts_l[s_idx:e_idx]
            # label_l_cut = label_l[s_idx:e_idx]
            label_l_cut = n_label[src_l_cut]
            size = len(src_l_cut)

            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)            
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            loss += clf_criterion_eval(lr_prob, src_label).item()
            pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()
    
    pred_label = pred_prob > 0.5
    auc_roc = roc_auc_score(n_label[src_l], pred_prob)
    val_acc = (pred_label == n_label[src_l]).mean()
    val_ap = average_precision_score(n_label[src_l], pred_prob)
    # val_f1 = f1_score(label_l, pred_prob)
    val_auc = roc_auc_score(n_label[src_l], pred_prob)
    return np.around(val_acc, 4), np.around(val_ap, 4), \
            np.around(val_f1, 4), np.around(val_auc, 4), np.around(loss / num_batch, 4)

def eval_epoch_reg(src_l, dst_l, ts_l, n_label, batch_size, lr_model, tgan, num_layer=NODE_LAYER):
    pred_prob = np.zeros(len(src_l))
    pred_label = np.zeros(len(src_l))
    loss = 0
    num_instance = len(src_l)
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
        lr_model.eval()
        tgan.eval()
        for k in range(num_batch):          
            s_idx = k * batch_size
            e_idx = min(num_instance - 1, s_idx + batch_size)
            src_l_cut = src_l[s_idx:e_idx]
            # dst_l_cut = dst_l[s_idx:e_idx]
            ts_l_cut = ts_l[s_idx:e_idx]
            # label_l_cut = label_l[s_idx:e_idx]
            label_l_cut = n_label[src_l_cut]
            size = len(src_l_cut)

            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)            
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            loss += clf_criterion_eval(lr_prob, src_label).item()
            pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()
            pred_label[s_idx:e_idx] = label_l_cut
    
    mae = mean_absolute_error(pred_label, pred_prob)
    return mae, np.around(loss / num_batch, 4)

# def eval_fairness(train_src_l, train_ts_l, Sn_feat, tgan, lr_model, penalty):
#     DP_sum = 0
#     num_test_instance = len(train_src_l)
#     num_test_batch = math.ceil(num_test_instance / BATCH_SIZE)
#     with torch.no_grad():
#         lr_model.eval()
#         for k in range(num_test_batch):
#             s_idx = k * BATCH_SIZE
#             e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
#             src_l_cut = train_src_l[s_idx:e_idx]
#             ts_l_cut = train_ts_l[s_idx:e_idx]
#             # label_l_cut = n_label[src_l_cut]
#             Sn_feat_cut = Sn_feat[src_l_cut]
            
#             with torch.no_grad():
#                 src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
            
#             src_Sn_feat = torch.from_numpy(Sn_feat_cut).float().to(device)
#             # src_Sn_feat = Sn_feat_cut

#             lr_prob = lr_model(src_embed).sigmoid()
#             DP_sum += penalty(lr_prob, src_Sn_feat).item()
    
#     DP = DP_sum / num_batch
#     return DP 

def eval_fairness(train_src_l, train_ts_l, Sn_feat, tgan, lr_model, penalty):
    pred_prob = torch.zeros(len(train_src_l)).to(device)
    Sn_feat_cum = torch.zeros(len(train_src_l)).to(device)
    num_test_instance = len(train_src_l)
    num_test_batch = math.ceil(num_test_instance / BATCH_SIZE)
    with torch.no_grad():
        lr_model.eval()
        for k in range(num_test_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = train_src_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            Sn_feat_cut = Sn_feat[src_l_cut]
            
            with torch.no_grad():
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
            
            src_Sn_feat = torch.from_numpy(Sn_feat_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            pred_prob[s_idx:e_idx] = lr_prob
            Sn_feat_cum[s_idx:e_idx] = src_Sn_feat

    DP = penalty(pred_prob, Sn_feat_cum).item()
    
    return DP*100


performances = []
fairnesss = []
device = torch.device('cuda:{}'.format(GPU))
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

test_sol = 1e-3
x_appro = torch.arange(test_sol, 1-test_sol, test_sol).to(device)
KDE_FAIR = kde_fair(x_appro)
penalty = KDE_FAIR.forward_dp

for run_time in range(RUNNING_TIME):
    start_time = time.time()
    random.seed(run_time)

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if args.clf=='clf':
        log_path = 'log/clf/'
    else:
        log_path = 'log/reg/'
    if args.sens_bn=='yes':
        log_path += 'adv_bn/'
    else:
        log_path += 'adv/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    fh = logging.FileHandler(log_path + f'{args.agg_method}-{args.attn_mode}-hyper={hyper_pent}-{run_time}.log', mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    valid_train_flag = (ts_l <= test_time)  
    valid_val_flag = (ts_l <= test_time) 
    assignment = np.random.randint(0, 10, len(valid_train_flag))
    valid_train_flag *= (assignment >= 2)
    valid_val_flag *= (assignment < 2)
    valid_test_flag = ts_l > test_time

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    # train_label_l = label_l[valid_train_flag]

    # use the validation
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    val_e_idx_l = e_idx_l[valid_val_flag]
    # val_label_l = label_l[valid_val_flag]

    # use the true test dataset
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    # test_label_l = label_l[valid_test_flag]


    ### Initialize the data structure for graph and edge sampling
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

    # full graph with all the data for the test and validation purpose
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

    ### Model initialize
    tgan = TGAN(train_ngh_finder, n_feat, e_feat,
                num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
    # optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
    # criterion = torch.nn.BCELoss()
    tgan = tgan.to(device)


    num_instance = len(train_src_l)
    # print(f'num_instance={num_instance}')
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    logger.debug('num of training instances: {}'.format(num_instance))
    logger.debug('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list) 


    clf = LR(n_feat.shape[1])
    clf_optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr)
    clf = clf.to(device)
    tgan.ngh_finder = full_ngh_finder
    idx_list = np.arange(len(train_src_l))
    
    if args.clf=='clf':
        clf_criterion = torch.nn.BCELoss()
        clf_criterion_eval = torch.nn.BCELoss()
    else:
        clf_criterion = torch.nn.MSELoss()
        clf_criterion_eval = torch.nn.MSELoss()
    
    if args.sens_bn=='yes':
        pretrain_Sn_feat = SHn_feat
    else:
        pretrain_Sn_feat = Sn_feat
    ### pretrain classifier
    N_CLF_EPOCHS = 10
    clf = pretrain_classifier(idx_list, train_src_l, train_ts_l, n_label, pretrain_Sn_feat, \
                        clf, tgan, clf_optimizer, clf_criterion, N_CLF_EPOCHS)

    ### pretrain adversary
    adv = Adversary(1, 1).to(device)
    adv_criterion = nn.MSELoss()
    adv_optimizer = optim.Adam(adv.parameters(), lr=args.lr)
    N_ADV_EPOCHS = 30
    adv = pretrain_adversary(idx_list, train_src_l, train_ts_l, n_label, pretrain_Sn_feat, \
                    clf, adv, tgan, adv_optimizer, adv_criterion, N_ADV_EPOCHS)

    t_total = time.time()

    for epoch in tqdm(range(args.n_epoch)):
        tgan = tgan.eval()

        ### train adv
        clf = clf.eval()
        adv = adv.train()
        np.random.shuffle(idx_list)

        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = train_src_l[s_idx:e_idx]
            dst_l_cut = train_dst_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            # label_l_cut = train_label_l[s_idx:e_idx]
            label_l_cut = n_label[src_l_cut]
            Sn_feat_cut = Sn_feat[src_l_cut]
            SHn_feat_cut = SHn_feat[src_l_cut]
            
            adv_optimizer.zero_grad()
            with torch.no_grad():
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
            
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            src_Sn_feat = torch.from_numpy(Sn_feat_cut).float().to(device)
            src_SHn_feat = torch.from_numpy(SHn_feat_cut).float().to(device)

            lr_prob = clf(src_embed).sigmoid()
            lr_prob = torch.unsqueeze(lr_prob, 1)
            p_z = adv(lr_prob).flatten()

            if args.sens_bn=='yes':
                adv_loss = adv_criterion(p_z, src_Sn_feat) * args.hyper_pent
            else:
                adv_loss = adv_criterion(p_z, src_SHn_feat) * args.hyper_pent
            adv_loss.backward()
            adv_optimizer.step()
        
        ### train classifier
        clf.train()
        adv.eval()
        np.random.shuffle(idx_list)
        num_batch_classiifer = 10
        for k in range(num_batch):
            if k>num_batch_classiifer:
                break
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = train_src_l[s_idx:e_idx]
            dst_l_cut = train_dst_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            # label_l_cut = train_label_l[s_idx:e_idx]
            label_l_cut = n_label[src_l_cut]
            Sn_feat_cut = Sn_feat[src_l_cut]
            SHn_feat_cut = SHn_feat[src_l_cut]
            
            clf_optimizer.zero_grad()
            with torch.no_grad():
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
            
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            src_Sn_feat = torch.from_numpy(Sn_feat_cut).float().to(device)
            src_SHn_feat = torch.from_numpy(SHn_feat_cut).float().to(device)

            lr_prob = clf(src_embed).sigmoid()
            lr_prob2 = torch.unsqueeze(lr_prob, 1)
            p_z = adv(lr_prob2).flatten()

            clf_loss = clf_criterion(lr_prob, src_label)
            if args.sens_bn=='yes':
                clf_loss -= adv_criterion(p_z, src_Sn_feat) * args.hyper_pent
            else:
                clf_loss -= adv_criterion(p_z, src_SHn_feat) * args.hyper_pent
            clf_loss.backward()
            clf_optimizer.step()
        
        ### evaluate performance and fairness
        ## fairness metrics
        D_SP = eval_fairness(train_src_l, train_ts_l, Sn_feat, tgan, clf, penalty)
        val_D_SP = eval_fairness(val_src_l, val_ts_l, Sn_feat, tgan, clf, penalty)
        test_D_SP = eval_fairness(test_src_l, test_ts_l, Sn_feat, tgan, clf, penalty)

        if args.clf=='clf':
            train_acc, train_ap, train_f1, \
            train_auc, train_loss = eval_epoch_clf(train_src_l, train_dst_l, \
                                                train_ts_l, n_label, BATCH_SIZE, clf, tgan)
            val_acc, val_ap, val_f1, \
            val_auc, val_loss = eval_epoch_clf(val_src_l, val_dst_l, val_ts_l, \
                                                n_label, BATCH_SIZE, clf, tgan)
            test_acc, test_ap, test_f1, \
            test_auc, test_loss = eval_epoch_clf(test_src_l, test_dst_l, test_ts_l, \
                                                n_label, BATCH_SIZE, clf, tgan)

            logger.info('epoch: {}:'.format(epoch))
            logger.info(f'train acc: {train_acc:.4f}, val acc: {val_acc:.4f}, test acc: {test_acc:.4f}')
            logger.info(f'train ap: {train_ap:.4f}, val ap: {val_ap:.4f}, test ap: {test_ap:.4f}')
            logger.info(f'train auc: {train_auc:.4f}, val auc: {val_auc:.4f}, test auc: {test_auc:.4f}')
        else:
            train_mae, train_loss = eval_epoch_reg(train_src_l, train_dst_l, \
                                                train_ts_l, n_label, BATCH_SIZE, clf, tgan)
            val_mae, val_loss = eval_epoch_reg(val_src_l, val_dst_l, val_ts_l, \
                                                n_label, BATCH_SIZE, clf, tgan)
            test_mae, test_loss = eval_epoch_reg(test_src_l, test_dst_l, test_ts_l, \
                                                n_label, BATCH_SIZE, clf, tgan)

            logger.info('epoch: {}:'.format(epoch))
            logger.info(f'train mae: {train_mae:.4f}, val mae: {val_mae:.4f}, test mae: {test_mae:.4f}')
            logger.info(f'train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, test loss: {test_loss:.4f}')


        logger.info('D_SP: {:.4f}, val D_SP: {:.4f}, test D_SP: {:.4f}'\
                    .format(D_SP, val_D_SP, test_D_SP))

    if args.clf=='clf':
        test_acc, test_ap, test_f1, \
        test_auc, test_loss = eval_epoch_clf(test_src_l, test_dst_l, test_ts_l, \
                                                    n_label, BATCH_SIZE, clf, tgan)
    else:
        test_mae, test_loss = eval_epoch_reg(test_src_l, test_dst_l, test_ts_l, \
                                                n_label, BATCH_SIZE, clf, tgan)
    #torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
    test_D_SP = eval_fairness(test_src_l, test_ts_l, Sn_feat, tgan, clf, penalty)
    print('============performace on test set=============')
    if args.clf=='clf':
        logger.info(f'test acc: {test_acc:.4f}, test ap: {test_ap:.4f}, test auc: {test_auc:.4f}')
    else:
        logger.info(f'test mae: {test_mae:.4f}, test loss: {test_loss:.4f}')
    logger.info('test D_SP: {:.4f}'.format(test_D_SP))
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
 
    ## record performance and fairness metrics
    if args.clf=='clf':
        performances.append([test_acc, test_auc, test_ap])
    else:
        performances.append([test_mae, test_loss])
    fairnesss.append([test_D_SP])

    print(f'running time={time.time() - start_time}')
    if run_time < RUNNING_TIME - 1:
        fh.close()
        logger.removeHandler(fh)

performance_mean = np.around(np.mean(performances, 0), 4)
performance_std = np.around(np.std(performances, 0), 4)
fairness_mean = np.around(np.mean(fairnesss, 0), 4)
fairness_std = np.around(np.std(fairnesss, 0), 4)

logger.info('Average of performance and fairness metric')
if args.clf=='clf':
    logger.info("Test statistics: -- acc: {:.4f}+-{:.4f} , auc: {:.4f}+-{:.4f}, ap: {:.4f}+-{:.4f}" \
                .format(performance_mean[0], performance_std[0], 
                        performance_mean[1], performance_std[1],
                        performance_mean[2], performance_std[2]))
else:
    logger.info("Test statistics: -- mae: {:.4f}+-{:.4f} , loss: {:.4f}+-{:.4f}" \
                .format(performance_mean[0], performance_std[0], 
                        performance_mean[1], performance_std[1]))

logger.info('Test statistics: -- D_SP: {:.4f}+-{}'\
            .format(fairness_mean[0], fairness_std[0]))
fh.close()
logger.removeHandler(fh)



 




