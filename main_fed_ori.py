#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
import numpy as np
import traceback
from tqdm import tqdm

from Algorithm.Training_Decoupled import Decoupled
from Algorithm.Training_ScaleFL import ScaleFL
from models.transformer import Transformer

# matplotlib.use('Agg')
import copy

from utils.Clients import Clients
from utils.options import args_parser
from models import *
from utils.get_datatset_new import get_dataset
from utils.utils import save_result
from utils.set_seed import set_random_seed
from Algorithm.Training_AdaptiveFL import AdaptiveFL
from Algorithm.Training_HeteroFL import HeteroFL

import sys
import logging
from datetime import datetime
# =============================================================================
# Tee logger: print to terminal + file
# =============================================================================
class TeeLogger:
    """Write stdout to both console and a file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def f2s(x: float) -> str:
    # 0.6 -> "0p6"  (避免文件名里出现多个点或奇怪格式)
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")

def FedAvg(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()
    print(net_glob)

    # training
    acc = []
    clients = Clients(args)
    for iter in tqdm(range(args.epochs)):  # tqdm 进度条库

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f"this epoch choose: {idxs_users}")
        for idx in idxs_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))

            clients.train(idx, iter)
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)


def FedProx(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedProx(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)
# 定义一个类用于重定向标准输出到日志文件
class RedirectStdoutToLog:
    def __init__(self, logger):
        self.logger = logger
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        if message.strip():  # 避免记录空消息
            self.logger.info(message.strip())

    def flush(self):
        pass
if __name__ == '__main__':
    args = args_parser()
    args.algorithm = 'AdaptiveFL'
    args.dataset = 'cifar100'
    args.model = 'resnet'
    args.num_classes = 100
    args.data_beta = 0.3
    args.iid = 0
    args.generate_data = 0
    args.epochs = 500
    args.num_users = 100
    args.frac = 0.1
    args.local_ep = 5
    args.local_bs = 50
    args.optimizer = 'sgd'
    args.lr = 0.01
    args.lr_decay = 0.998
    args.seed = 1
    args.client_chosen_mode = 'available'
    args.depth_saved = [2, 3, 4]
    args.width_ration = [0.4, 0.66, 1.0]
    args.result_dir = os.path.join('results', 'temporal_support_v2_beta085_full500')
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    os.makedirs(args.result_dir, exist_ok=True)
    log_path = os.path.join(args.result_dir, 'temporal_support_v2_beta085_full500.log')
    original_stdout = sys.stdout
    logger = TeeLogger(log_path)
    sys.stdout = logger
    started = datetime.now()
    try:
        print('TemporalSupport V2 formal experiment')
        print('beta=0.85; 500 rounds; Full-only evaluation; device={}'.format(args.device))
        print('profiles=[(0.4,2), (0.4,3), (0.4,4), (0.66,2), (0.66,3), '
              '(0.66,4), (1.0,2)]; BN track_running_stats=False')
        set_random_seed(args.seed)
        dataset_train, dataset_test, dict_users = get_dataset(args)
        client_sizes = [len(dict_users[index]) for index in range(args.num_users)]
        print('client sample sizes: min={}, median={}, max={}'.format(
            min(client_sizes), float(np.median(client_sizes)), max(client_sizes)))
        result = AdaptiveFL(args, dataset_train, dataset_test, dict_users)
        elapsed_seconds = (datetime.now() - started).total_seconds()
        print('completed 500/500 rounds; elapsed_seconds={:.3f}; average_round_seconds={:.3f}'.format(
            elapsed_seconds, elapsed_seconds / args.epochs))
        print('recorded_full_accuracy_rounds={}'.format(len(result['accuracy'])))
    except Exception:
        traceback.print_exc()
        raise
    finally:
        sys.stdout = original_stdout
        logger.close()
