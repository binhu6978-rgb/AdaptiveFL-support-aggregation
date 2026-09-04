#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time
import json
import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
from tqdm import tqdm

from models.Fed import (get_model_list, Aggregation_AdaptiveFL, Aggregation_Support,
                        split_model, select_clients)
from models.vgg import vgg_16_bn
from models.resnet import ResNet18_cifar
from models.resnet import ResNet18_widar
from utils.HeteroClients import HeteroClients
from models.test import test_img, test
from models.Update import DatasetSplit, LocalUpdate_AdaptiveFL
from optimizer.Adabelief import AdaBelief


def _assert_prefix_consistency(net_glob_list):
    largest_state = net_glob_list[-1].state_dict()
    for model_idx, net in enumerate(net_glob_list):
        expected_state = split_model(largest_state, net.state_dict())
        actual_state = net.state_dict()
        for key in actual_state:
            if not torch.equal(actual_state[key], expected_state[key]):
                max_abs_diff = (actual_state[key] - expected_state[key]).abs().max().item()
                raise AssertionError(
                    "Round 1 prefix inconsistency: model={}, parameter={}, max_abs_diff={}".format(
                        model_idx, key, max_abs_diff))
    print("Round 1 prefix consistency assertion: PASSED")


def AdaptiveFL(args, dataset_train, dataset_test, dict_users):
    net_glob_list, net_slim_info = get_model_list(args)

    full_idx = len(net_glob_list) - 1
    full_profile = net_slim_info[full_idx]
    if float(full_profile[0]) != 1.0 or int(full_profile[1]) != 2:
        raise AssertionError(
            "Expected Full profile (1.0, 2), got {}".format(full_profile))
    print("Full evaluation model: {}".format(full_profile))

    # training
    total_time = 0
    time_list = []
    full_acc = []
    clients = HeteroClients(args, net_slim_info)

    # 开始训练
    for iter in tqdm(range(args.epochs)):  # tqdm 进度条库

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        if iter == 1:
            _assert_prefix_consistency(net_glob_list)

        w_locals = []
        lens = []

        m = max(int(args.frac * args.num_users), 1)

        # 用户选择
        if args.client_chosen_mode == 'RL':  # RL-Selection
            # ration_users = np.random.choice([2, 5, 6], m)  # 模型选择
            ration_users = np.random.choice(range(len(net_glob_list)), m)  # 模型选择
            idx_users = clients.select_clients(ration_users)  # 基于强化学习的客户端选择
        elif args.client_chosen_mode == 'greedy':  # greedy 选择
            ration_users = np.random.choice([len(net_glob_list)-1], m)  # 模型选择
            idx_users = random.sample(range(args.num_users), len(ration_users))  # 基于强化学习的客户端选择
        else:
            ration_users = np.random.choice(range(len(net_glob_list)), m)  # 模型选择
            idx_users = select_clients(args, ration_users, len(net_glob_list))  # 基于规则的客户端选择

        feedback_list = []
        max_time = 0
        for id, idx in enumerate(idx_users):
            begin = time.time()
            local = LocalUpdate_AdaptiveFL(args=args, dataset=dataset_train, idxs=dict_users[idx])
            if args.client_chosen_mode == 'RL' or args.client_chosen_mode == 'random' or args.client_chosen_mode == 'greedy':  # 在客户端本地做自适应裁剪
                feedback_model_idx = clients.train(idx, ration_users[id])
            elif args.client_chosen_mode == 'available' or args.client_chosen_mode == 'fit':
                feedback_model_idx = ration_users[id]  # 这里是基于规则的，不会出错，分发的是什么模型，返回的就是什么模型
            feedback_list.append(feedback_model_idx)

            w = local.train(round=iter, net=copy.deepcopy(net_glob_list[feedback_model_idx]).to(args.device))  # 这里开始正式训练

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
            time_epoch = time.time() - begin
            max_time = max(max_time, time_epoch)

        total_time += max_time

        time_list.append(total_time)
        print(f"this epoch cost time:{max_time}")

        print(f"this epoch choose: {idx_users}")  # 这一轮选择的用户下标
        print(f"this epoch dispatch models: {ration_users}")  # 每个用户对应的模型的比例, 初始分发
        print(f"this epoch received models: {feedback_list}")  # 每个用户对应的模型的比例, 初始分发
        print(f"hetero_proportion: \t{args.client_hetero_ration}")
        # 需要print 每个客户端的计算资源

        if iter == 0:
            w_glob_param = Aggregation_AdaptiveFL(
                w_locals, lens, net_glob_list[-1].state_dict())
        else:
            w_glob_param = Aggregation_Support(
                w_locals, lens, net_glob_list[-1].state_dict(), epsilon=0.2)

        for net in net_glob_list:
            net.load_state_dict(split_model(w_glob_param, net.state_dict()))

        print("Full evaluation profile: {}".format(full_profile))
        full_acc.append(test(net_glob_list[full_idx], dataset_test, args))

    result_dir = getattr(args, 'result_dir', 'results/support_v1_eps02_full')
    os.makedirs(result_dir, exist_ok=True)
    accuracy_path = os.path.join(result_dir, 'full_accuracy.json')
    result = {
        'full_profile': {
            'width': float(full_profile[0]),
            'depth': int(full_profile[1]),
            'parameters_million': float(full_profile[2]),
        },
        'epsilon': 0.2,
        'rounds': len(full_acc),
        'accuracy': full_acc,
        'cumulative_max_client_train_time_seconds': time_list,
    }
    with open(accuracy_path, 'w', encoding='utf-8') as result_file:
        json.dump(result, result_file, indent=2)
    print("Saved Full-only accuracy: {}".format(accuracy_path))
    return full_acc


'''
                    
            for id, p in enumerate(model_proportion):
                if id == 0:  # small model
                    idx_users = np.random.choice(range(args.num_users), int(m * p), replace=False)
                    ration_users = [random.randint(0, 2) for _ in range(int(m * p))]
                elif id == 1:  # medium model
                    idx_users = np.hstack((idx_users,
                                           np.random.choice(
                                               [i for i in
                                                range(int(args.num_users * sum(hetero_proportion[:id])), args.num_users)  # 做个sum
                                                if i not in idx_users],
                                               int(m * p), replace=False)))
                    ration_users.extend([random.randint(3, 5) for _ in range(int(m * p))])
                elif id == 2:  # large model
                    idx_users = np.hstack((idx_users,
                                           np.random.choice(
                                               [i for i in
                                                range(int(args.num_users * sum(hetero_proportion[:id])), args.num_users)  # 做个sum
                                                if i not in idx_users],
                                               int(m * p), replace=False)))
                    ration_users.extend([6 for _ in range(int(m * p))])
'''
