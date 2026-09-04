#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import traceback
import matplotlib
import numpy as np
from tqdm import tqdm

from Algorithm.Training_Decoupled import Decoupled
from Algorithm.Training_ScaleFL import ScaleFL

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
# 重定向标准输出到日志
sys.stdout = RedirectStdoutToLog(logging.getLogger())
import os
if __name__ == '__main__':
    # parse args
    run_dir = 'results/all_large_oracle_300'
    os.makedirs(run_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # redirect stdout to log file (tee)


    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # algorithms = ['AdaptiveFL', 'HeteroFL', 'ScaleFL','FedAvg']
    algorithms = ['AdaptiveFL']

    for algorithm in algorithms:
        try:
            args.epochs = 300
            args.num_users = 100
            args.frac = 0.1
            args.local_ep = 5
            args.local_bs = 50
            args.optimizer = 'sgd'
            args.lr = 0.01
            args.lr_decay = 0.998
            args.seed = 1
            args.generate_data = 0
            args.data_beta = 0.3
            args.iid = 0
            args.client_chosen_mode = 'available'
            # args.epochs=5
            args.model = 'resnet'
            args.dataset = 'cifar100'
            args.algorithm = algorithm
            args.result_dir = run_dir

            if args.dataset.lower() == 'cifar10':
                args.num_classes = 10
            elif args.dataset.lower() == 'cifar100':
                args.num_classes = 100
            elif args.dataset.lower() in ['tinyimagenet', 'tiny_imagenet', 'tiny-imagenet']:
                args.num_classes = 200
            else:
                args.num_classes = 10  # 默认值，可根据需要修改
            log_path = os.path.join(run_dir, 'all_large_oracle_300round.log')

            
            logging.basicConfig(level=logging.DEBUG,
                                 format='%(asctime)s - %(levelname)s - %(message)s',
                                 filename=log_path,
                                 filemode='w')


            set_random_seed(args.seed)

            dataset_train, dataset_test, dict_users = get_dataset(args)
            client_sample_sizes = np.asarray([len(dict_users[idx]) for idx in range(args.num_users)])
            print("Client sample size: min={}, median={}, max={}".format(
                int(client_sample_sizes.min()),
                float(np.median(client_sample_sizes)),
                int(client_sample_sizes.max())))
            if (int(client_sample_sizes.min()), float(np.median(client_sample_sizes)),
                    int(client_sample_sizes.max())) == (500, 500.0, 500):
                raise AssertionError("IID-like 500/500/500 client split detected; expected cached non-IID split")
            print("Experiment log: {}".format(log_path))
            # plot_client_distribution(dataset_train, dict_users)

            if args.algorithm == 'AdaptiveFL':
                # AdaptiveFL constructs its model pool through get_model_list().
                net_glob = None
            elif args.model == 'cnn':
                if args.dataset == 'femnist':
                    net_glob = CNNFashionMnist(args)
                elif args.dataset == 'mnist':
                    net_glob = CNNMnist(args)
                elif args.use_project_head:
                    net_glob = ModelFedCon(args.model, args.out_dim, args.num_classes)
                elif 'cifar' in args.dataset:
                    net_glob = CNNCifar(args)
            elif 'resnet' in args.model:
                net_glob = ResNet18_cifar(num_channels=args.num_channels, num_classes=args.num_classes)  # 默认为3通道
            elif 'mobilenet' in args.model:
                net_glob = MobileNetV2(args.num_channels, args.num_classes)
            elif 'vgg' in args.model:
                net_glob = vgg_16_bn(num_classes=args.num_classes, track_running_stats=True, num_channels=args.num_channels)
            elif 'lstm' in args.model:
                net_glob = CharLSTM()
            elif 'transformer' in args.model:
                from models.transformer import Transformer
                net_glob = Transformer(vocab_size=30522, d_model=128, nhead=8, num_encoder_layers=8, max_len=256,
                                    slim_idx=2,
                                    scale=0.5, dropout=0.1)
            if args.algorithm == 'FedAvg':
                net_glob.to(args.device)
                FedAvg(net_glob, dataset_train, dataset_test, dict_users)
            elif args.algorithm == 'FedProx':
                FedProx(net_glob, dataset_train, dataset_test, dict_users)
            elif args.algorithm == 'AdaptiveFL':
                args.depth_saved = [2]
                args.width_ration = [1.0]
                experiment_start = time.time()
                AdaptiveFL(args, dataset_train, dataset_test, dict_users)
                elapsed = time.time() - experiment_start
                print("Total experiment wall time (seconds): {:.6f}".format(elapsed))
                print("Average wall time per round (seconds): {:.6f}".format(
                    elapsed / args.epochs))
            elif args.algorithm == 'HeteroFL':
                args.depth_saved = [0]
                args.width_ration = [0.5, 0.71, 1.0]
                HeteroFL(args, dataset_train, dataset_test, dict_users)
            elif args.algorithm == 'ScaleFL':
                args.depth_saved = [2, 3, 4]
                args.width_ration = [0.4, 0.66, 1.0]
                ScaleFL(args, dataset_train, dataset_test, dict_users)
            elif args.algorithm == 'Decoupled':
                args.depth_saved = [8]
                args.width_ration = [0.4, 0.66, 1.0]
                Decoupled(args, dataset_train, dataset_test, dict_users)
            else:
                raise "%s algorithm has not achieved".format(args.algorithm)

        except Exception:
            traceback.print_exc(file=sys.stdout)
            raise
