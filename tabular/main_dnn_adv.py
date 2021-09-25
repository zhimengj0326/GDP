import sys, os
# sys.path.append(os.path.abspath(os.path.join('../..')))

import torch
import numpy as np
import torch.utils.data as data_utils
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from data_load import read_dataset
from kde import kde_fair

from sklearn.metrics import accuracy_score,roc_auc_score,recall_score
from sklearn.metrics import average_precision_score
import sys
import random
import argparse
import logging
import time

class NetRegression(nn.Module):
    def __init__(self, input_size, num_classes, size = 50):
        super(NetRegression, self).__init__()
        self.first = nn.Linear(input_size, size)
        self.fc = nn.Linear(size, size)
        self.last = nn.Linear(size, num_classes)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        out = F.selu(self.first(x))
        out = F.selu(self.fc(out))
        out = self.last(out)
        return self.sigmoid(out)

def pretrain_classifier(clf, data_loader, optimizer, criterion):
    for x, y, _ in data_loader:
        clf.zero_grad()
        p_y = clf(x).flatten()
        # print(f'y={y}')
        # print(f'p_y={p_y}')
        loss = criterion(p_y, y)
        # loss = Variable(loss, requires_grad = True)
        loss.backward()
        optimizer.step()
    return clf

class Adversary(nn.Module):

    def __init__(self, n_sensitive, n_hidden=32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


def pretrain_adversary(adv, clf, data_loader, optimizer, criterion):
    for x, _, z in data_loader:
        p_y = clf(x)
        adv.zero_grad()
        # print(f'p_y={p_y}')
        p_z = adv(p_y).flatten()
        # print(f'p_z={p_z}')
        # print(f'z={z}')
        loss = criterion(p_z, z) * penalty_coefficient
        loss.backward()
        optimizer.step()
    return adv

def regularized_learning(dataset_loader, x_train, y_train, z_train, x_test, y_test, z_test, \
                        clf, adv, device_gpu, logger, penalty_coefficient, \
                        clf_criterion, adv_criterion, fairness_penalty, clf_optimizer, \
                        adv_optimizer, num_epochs):

    # mse regression objective
    # data_fitting_loss = nn.MSELoss()

    for j in range(num_epochs):
        for i, (x, y, z) in enumerate(dataset_loader):
            outputs = clf(x)
            adv.zero_grad()
            p_z = adv(outputs).flatten()

            ### train adv
            loss_adv = adv_criterion(p_z, z) * penalty_coefficient
            loss_adv.backward()
            adv_optimizer.step()

        num_batches_clf = 3
        for i, (x, y, z) in enumerate(dataset_loader):
            if i<num_batches_clf:
                outputs = clf(x)
                clf.zero_grad()
                p_z = adv(outputs).flatten()
                ### train clf
                loss_adv = adv_criterion(p_z, z) * penalty_coefficient
                clf_loss = clf_criterion(outputs.flatten(), y) - loss_adv

                clf_loss.backward()
                clf_optimizer.step()

        if dataset_name == 'crimes':
            loss_train, loss_test, mae_train, mae_test, DP_train, DP_test = evaluate(clf, clf_criterion, fairness_penalty, \
                                    x_train, y_train, z_train, x_test, y_test, z_test, device_gpu)
            logger.info('epoch: {}:'.format(j))
            logger.info(f'train loss: {loss_train:.4f}, train mae: {mae_train:.4f}, DP train: {DP_train:.4f}')
            logger.info(f'test loss: {loss_test:.4f}, test mae: {mae_test:.4f}, DP test: {DP_test:.4f}')
        else:
            loss_train, loss_test, acc_train, acc_test, DP_train, DP_test = evaluate(clf, clf_criterion, fairness_penalty, \
                                    x_train, y_train, z_train, x_test, y_test, z_test, device_gpu)
            logger.info('epoch: {}:'.format(j))
            logger.info(f'train loss: {loss_train:.4f}, train acc: {acc_train:.4f}, DP train: {DP_train:.4f}')
            logger.info(f'test loss: {loss_test:.4f}, test acc: {acc_test:.4f}, DP test: {DP_test:.4f}')

def evaluate_reg(model, data_fitting_loss, fairness_penalty, x_train, y_train, z_train, x, y, z, device_gpu):
    prediction = model(x_train).detach().flatten()
    loss_train = data_fitting_loss(prediction, y_train).item()
    mae_train = nn.L1Loss()(prediction, y_train).item()
    DP_train = fairness_penalty(prediction, z_train, device_gpu).item()


    prediction = model(x).detach().flatten()
    loss_test = data_fitting_loss(prediction, y).item()
    mae_test = nn.L1Loss()(prediction, y).item()
    DP_test = fairness_penalty(prediction, z, device_gpu).item()
    return loss_train, loss_test, mae_train, mae_test, DP_train, DP_test

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0.5).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct.item() / len(labels)
    return acc

def evaluate_class(model, data_fitting_loss, fairness_penalty, x_train, y_train, z_train, x, y, z, device_gpu):
    prediction = model(x_train).detach().flatten()
    loss_train = data_fitting_loss(prediction, y_train).item()
    DP_train = fairness_penalty(prediction, z_train, device_gpu).item()

    acc_train = accuracy(prediction, y_train)

    prediction = model(x).detach().flatten()
    loss_test = data_fitting_loss(prediction, y).item()

    acc_test = accuracy(prediction, y)

    DP_test = fairness_penalty(prediction, z, device_gpu).item()
    return loss_train, loss_test, acc_train, acc_test, DP_train, DP_test

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('--data', type=str, help='data sources to use', default='crimes')
parser.add_argument('--batch_size', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--times', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--check_sol', type=float, default=1e-3, help='check solution')
parser.add_argument('--hyper_pent', type=float, default=1.0, help='Hyperparmeters for penalty')
parser.add_argument('--sens_bn', type=bool, default=False, help='Binary sensitive attribute')
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

batch_size = args.batch_size
GPU = args.gpu
num_epochs = args.n_epoch
lr = args.lr
test_sol = args.check_sol
penalty_coefficient = args.hyper_pent
dataset_name = args.data
RUNNING_TIME = args.times

x_train, y_train, z_train, x_test, y_test, z_test = read_dataset(name=dataset_name, fold=1)
n, d = x_train.shape


device_gpu = torch.device('cuda:{}'.format(GPU))

if dataset_name == 'crimes':
    data_fitting_loss = nn.MSELoss()
    evaluate = evaluate_reg
else:
    data_fitting_loss = nn.functional.binary_cross_entropy  ##nn.CrossEntropyLoss()
    evaluate = evaluate_class

# num_epochs = 20
# lr = 1e-5

# penalty_coefficient = 1.0
# test_sol = 1e-3
x_appro = torch.arange(test_sol, 1-test_sol, test_sol).to(device_gpu)
KDE_FAIR = kde_fair(x_appro)
penalty = KDE_FAIR.forward

# wrap dataset in torch tensors
# print(f'y_train={y_train}')
y_train = torch.tensor(y_train.astype(np.float32)).to(device_gpu)
x_train = torch.tensor(x_train.astype(np.float32)).to(device_gpu)
z_train = torch.tensor(z_train.astype(np.float32)).to(device_gpu)
z_train_bn = (z_train>0.5).float()

if args.sens_bn:
    dataset = data_utils.TensorDataset(x_train, y_train, z_train_bn)
else:   
    dataset = data_utils.TensorDataset(x_train, y_train, z_train)
dataset_loader = data_utils.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
dataset_loader = data_utils.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# print(f'x_test={type(x_test)}')

y_test = torch.tensor(y_test.astype(np.float32)).to(device_gpu)
x_test = torch.tensor(x_test.astype(np.float32)).to(device_gpu)
z_test = torch.tensor(z_test.astype(np.float32)).to(device_gpu)

performances = []
fairnesss = []
for run_time in range(RUNNING_TIME):
    t_total = time.time()
    # run_time = 0
    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if args.sens_bn:
        log_path = f'log/{dataset_name}/adv_bn'
    else:
        log_path = f'log/{dataset_name}/adv'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    fh = logging.FileHandler(log_path + f'/{args.prefix}-hyper={penalty_coefficient}-{run_time}.log', mode='w')

    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    ### pretrain classifier
    clf = NetRegression(d, 1).to(device_gpu)
    clf_criterion = data_fitting_loss
    clf_optimizer = optim.Adam(clf.parameters(), lr=lr)
    N_CLF_EPOCHS = 20

    for epoch in range(N_CLF_EPOCHS):
        clf = pretrain_classifier(clf, dataset_loader, clf_optimizer, clf_criterion)

    ### pretrain adversary
    # lambdas = torch.Tensor([130, 30])
    adv = Adversary(1).to(device_gpu)
    adv_criterion = nn.MSELoss()
    adv_optimizer = optim.Adam(adv.parameters(), lr=lr)
    N_ADV_EPOCHS = 60

    for epoch in range(N_ADV_EPOCHS):
        pretrain_adversary(adv, clf, dataset_loader, adv_optimizer, adv_criterion)

    regularized_learning(dataset_loader, x_train, y_train, z_train, x_test, y_test, z_test, \
                        clf, adv, device_gpu, logger, penalty_coefficient, \
                        clf_criterion, adv_criterion, penalty, clf_optimizer, \
                        adv_optimizer, num_epochs)
    if dataset_name == 'crimes':
        loss_train, loss_test, mae_train, mae_test, DP_train, DP_test = evaluate(clf, data_fitting_loss, penalty, \
                                    x_train, y_train, z_train, x_test, y_test, z_test, device_gpu)
        print(f'loss train: {loss_train:.4f}, mae train: {mae_train:.4f}, DP train: {DP_train:.4f}')
        print(f'loss test: {loss_test:.4f}, mae test: {mae_test:.4f}, DP test: {DP_test:.4f}')
        ## record performance and fairness metrics
        performances.append([loss_train, loss_test, mae_train, mae_test])
        fairnesss.append([DP_train, DP_test])
    else:
        loss_train, loss_test, acc_train, acc_test, DP_train, DP_test = evaluate(clf, data_fitting_loss, penalty, \
                                    x_train, y_train, z_train, x_test, y_test, z_test, device_gpu)
        print(f'loss train: {loss_train:.4f}, acc train: {acc_train:.4f}, DP train: {DP_train:.4f}')
        print(f'loss test: {loss_test:.4f}, acc test: {acc_test:.4f}, DP test: {DP_test:.4f}')
        ## record performance and fairness metrics
        performances.append([loss_train, loss_test, acc_train, acc_test])
        fairnesss.append([DP_train, DP_test])

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
logger.info("Test statistics: -- train loss: {:.4f}+-{:.4f} , test loss: {:.4f}+-{:.4f}" \
            .format(performance_mean[0], performance_std[0], 
                    performance_mean[1], performance_std[1]))
if dataset_name == 'crimes':
    logger.info("Test statistics: -- train mae: {:.4f}+-{:.4f} , test mae: {:.4f}+-{:.4f}" \
                .format(performance_mean[2], performance_std[2], 
                        performance_mean[3], performance_std[3]))
else:
    logger.info("Test statistics: -- train acc: {:.4f}+-{:.4f} , test acc: {:.4f}+-{:.4f}" \
                .format(performance_mean[2], performance_std[2], 
                        performance_mean[3], performance_std[3]))

logger.info('Test statistics: -- train D_SP: {:.4f}+-{:.4f}, test D_SP: {:.4f}+-{:.4f}'\
            .format(fairness_mean[0], fairness_std[0], 
                    fairness_mean[1], fairness_std[1]))
fh.close()
logger.removeHandler(fh)