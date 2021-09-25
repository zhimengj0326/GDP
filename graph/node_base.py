import time
import argparse
import numpy as np
import logging
import time
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score,roc_auc_score,recall_score
from sklearn.metrics import average_precision_score

from utils import load_data, accuracy,load_pokec
from module import GAT, GCN, SGC

from kde import kde_fair

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='assigned gpu.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--model', type=str, default="GAT",
                    help='the type of model GCN/GAT')
parser.add_argument('--dataset', type=str, default='pokec_n',
                    choices=['pokec_z','pokec_n'])
parser.add_argument('--num-hidden', type=int, default=64,
                    help='Number of hidden units of classifier.')
parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=.6,
                    help="attention dropout")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument("--bias", action='store_true', default=False,
            help="flag to use bias")
parser.add_argument('--acc', type=float, default=0.5,
                    help='the selected FairGNN accuracy on val would be at least this high')
parser.add_argument('--roc', type=float, default=0.5,
                    help='the selected FairGNN ROC score on val would be at least this high')
parser.add_argument('--running_times', type=int, default=5, help='number of running times')
parser.add_argument('--hyper', type=float, default=0., help="hyperparameter for penality")
parser.add_argument('--sens_bn', type=bool, default=False, help='Binary sensitive attribute')

args = parser.parse_args()

RUNNING_TIME = args.running_times
hyper = args.hyper

device = torch.device('cuda:{}'.format(args.gpu))
print(args)
#%%
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
print(args.dataset)


if args.dataset == 'pokec_z':
    dataset = 'region_job'
else:
    dataset = 'region_job_2'
sens_attr = "AGE"
predict_attr = "spoken_languages_indicator"

seed = 20
path="/data/zhimengj/dataset/pokec/"
test_idx=False

adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_pokec(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,
                                                                                    path=path,
                                                                                    seed=seed,test_idx=test_idx)
print(len(idx_test))
#%%
import dgl
from utils import feature_norm
# g = dgl.DGLGraph()
g = dgl.from_scipy(adj)
g = g.to(device)


n_classes = 2

min_sens, max_sens= torch.min(sens), torch.max(sens)
sens = (sens - min_sens) / (max_sens - min_sens)

# print(f'features={features.shape}')

labels[labels>1]=1
# if sens_attr:
#     sens[sens>0]=1

# add self loop
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
n_edges = g.number_of_edges()

# model = FairGNN(nfeat = features.shape[1], args = args)
# model.estimator.load_state_dict(torch.load("./checkpoint/GCN_sens_{}_ns_{}".format(dataset,sens_number)))


features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)
sens = sens.to(device)
if args.sens_bn:
    sens_train = (sens>0.5).float()
else:
    sens_train = sens
idx_sens_train = idx_sens_train.to(device)

test_sol = 1e-3
x_appro = torch.arange(test_sol, 1-test_sol, test_sol).to(device)
KDE_FAIR = kde_fair(x_appro)
penalty = KDE_FAIR.forward_dp


performances = []
fairnesss = []
for run_time in range(RUNNING_TIME):
    
    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    log_path = f'log/{args.dataset}/{args.prefix}/{args.model}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    fh = logging.FileHandler(log_path + f'/hyper={hyper}-{run_time}.log', mode='w')

    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    # Model and optimizer
    if args.model=="GAT":
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT(g,
                    args.num_layers,
                    features.shape[1],
                    args.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual)

    elif args.model=="GCN":
        model = GCN(g,
                    features.shape[1],
                    args.num_hidden,
                    n_classes,
                    args.num_layers,
                    F.relu,
                    args.dropout)
    elif args.model=="SGC":
        power_k = 2
        model = SGC(g,
                    features.shape[1],
                    n_classes,
                    args.num_hidden,
                    power_k)
    model = model.to(device)

    # Train model
    t_total = time.time()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        t = time.time()
        ### inference
        # train_features = features[idx_train]
        model.train()
        train_labels = labels[idx_train]
        all_h,all_logit = model(features)
        all_y = F.softmax(all_logit, dim=1)

        ### training loss
        cls_loss = criterion(all_logit[idx_train],train_labels.long())
        cls_loss += hyper * penalty(all_y[idx_train, 1], sens_train[idx_train])

        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        model.eval()
        # all_h,all_y = model(features)
        all_y = F.softmax(all_logit, dim=1)
        # print(f'all_y={all_y}')
        # print(f'labels={labels}')
        acc_train = accuracy(all_y[idx_train, 1], labels[idx_train]).item()
        ap_train = average_precision_score(labels[idx_train].cpu().numpy(), all_y[idx_train, 1].detach().cpu().numpy())
        roc_train = roc_auc_score(labels[idx_train].cpu().numpy(),all_y[idx_train, 1].detach().cpu().numpy())

        parity_train = penalty(all_y[idx_train, 1], sens[idx_train]).item()

        acc_val = accuracy(all_y[idx_val, 1], labels[idx_val]).item()
        ap_val = average_precision_score(labels[idx_val].cpu().numpy(), all_y[idx_val, 1].detach().cpu().numpy())
        roc_val = roc_auc_score(labels[idx_val].cpu().numpy(),all_y[idx_val, 1].detach().cpu().numpy())

        
        parity_val = penalty(all_y[idx_val, 1], sens[idx_val]).item()

        acc_test = accuracy(all_y[idx_test, 1], labels[idx_test]).item()
        ap_test = average_precision_score(labels[idx_test].cpu().numpy(), all_y[idx_test, 1].detach().cpu().numpy())
        roc_test = roc_auc_score(labels[idx_test].cpu().numpy(),all_y[idx_test, 1].detach().cpu().numpy())


        parity = penalty(all_y[idx_test, 1], sens[idx_test]).item()

        logger.info('epoch: {}:'.format(epoch))
        logger.info(f'train acc: {acc_train:.4f}, val acc: {acc_val:.4f}, test acc: {acc_test:.4f}')
        logger.info(f'train ap: {ap_train:.4f}, val ap: {ap_val:.4f}, test ap: {ap_test:.4f}')
        # logger.info(f'train f1: {train_f1}, test f1: {val_f1}')
        logger.info(f'train auc: {roc_train:.4f}, val auc: {roc_val:.4f}, test auc: {roc_test:.4f}')
        logger.info('D_SP: {:.4f}, val D_SP: {:.4f}, test D_SP: {:.4f}'\
                    .format(parity_train, parity_val, parity))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print('============performace on test set=============')

    logger.info(f'test acc: {acc_test:.4f}, test ap: {ap_test:.4f}, test auc: {roc_test:.4f}')
    logger.info('test D_SP: {:.4f}'.format(parity))
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    ## record performance and fairness metrics
    performances.append([acc_test, roc_test, ap_test])
    fairnesss.append([parity])

    print(f'running time={time.time() - t_total}')
    if run_time < RUNNING_TIME - 1:
        fh.close()
        logger.removeHandler(fh)

### statistical results
performance_mean = np.around(np.mean(performances, 0), 4)
performance_std = np.around(np.std(performances, 0), 4)
fairness_mean = np.around(np.mean(fairnesss, 0), 4)
fairness_std = np.around(np.std(fairnesss, 0), 4)

logger.info('Average of performance and fairness metric')
logger.info("Test statistics: -- acc: {:.4f}+-{:.4f} , auc: {:.4f}+-{:.4f}, ap: {:.4f}+-{:.4f}" \
            .format(performance_mean[0], performance_std[0], 
                    performance_mean[1], performance_std[1],
                    performance_mean[2], performance_std[2]))
logger.info('Test statistics: -- D_SP: {:.4f}+-{:.4f}'\
            .format(fairness_mean[0], fairness_std[0]))
fh.close()
logger.removeHandler(fh)