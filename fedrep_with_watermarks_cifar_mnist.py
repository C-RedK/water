# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg),
# FedAvg (--alg fedavg) and FedProx (--alg prox)

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
import os

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data,getdata
from models.Update import LocalUpdate
from models.test import test_img_local_all
from utils.watermark import get_X, get_b
from torch.backends import cudnn

import time
import random

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)


def main(args,rd,seed):
    # Step1:设置随机种子,保证结果可复现
    init_seed(seed=seed)

    # Step1：参数初始化
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Step2：客户数据集初始化
    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        # ck 修改get_data函数
        dataset_train, dataset_test, dict_users_train, dict_users_test = getdata(args)
    
    else:
        raise Exception

    # print("args.alg:{}".format(args.alg)) #tyl：和之前一样，统一打印需要观测的实验参数
    
    # Step3：模型初始化
    net_glob = get_model(args)
    net_glob.train()  # tyl:这个步骤原因？
    if args.load_fed != 'n': 
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    print('net_glob.state_dict().keys():')
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # Step4：水印信息初始化
    # build watermark----------------------------------------------------------------------------------------------
    dict_X = get_X(net_glob=net_glob, embed_dim=args.embed_dim, num_users=args.num_users,use_watermark=args.use_watermark)
    dict_b = get_b(embed_dim=args.embed_dim, num_users=args.num_users,use_watermark=args.use_watermark)
    # build watermark----------------------------------------------------------------------------------------------

    # Step5：表示层参数与全局模型解耦
    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    w_glob_keys = []
    if args.alg == 'fedrep' or args.alg == 'fedper':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 3, 4]]
        elif 'mnist' in args.dataset and args.model == 'cnn':
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 3, 4]]
        else:
            raise Exception
    if args.alg == 'fedavg':
        w_glob_keys = []
    if 'sent140' not in args.dataset:
        # 我他妈真醉了，真的服了,这里是把二维的变成一维的，然后后面的 in w_glob_keys才有用啊
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    print('total_num_layers:{}'.format(total_num_layers))
    print('w_glob_keys:{}'.format(w_glob_keys)) #tyl：表示层模型keys
    print('net_keys:{}'.format(net_keys)) # tyl：全局模型keys
    # tyl：这部分统计表示层参数总数和总的参数总数，后面似乎也没用到，考虑后面注释掉
    if args.alg == 'fedrep':
        num_param_glob = 0 # tyl：统计表示层参数个数
        num_param_local = 0 #tyl：统计所有层参数个数
        for key in net_glob.state_dict().keys():
            # tensor.numel() 统计tensor中元素的个数
            num_param_local += net_glob.state_dict()[key].numel()
            print(num_param_local)
            if key in w_glob_keys:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate:{}, batch size:{}".format(args.lr, args.local_bs))

    # Step6：Client模型参数初始化
    # generate list of local models for each user
    net_local_list = [] #tyl：没用到？
    # w_locals:客户端中拥有的参数
    # 我感觉下面这些代码是
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    # Step7：模型训练与水印嵌入
    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    success_rates = []
    start = time.time()
    all_one_for_all_clients_rates = []
    for iter in range(args.epochs + 1):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        # 最后一轮训练慢的原因是 最后一轮选择了全部客户端进行训练，所以才会这么慢
        # 原本可能选十几二十个，但是现在是所有的都他妈的要来训练，自然就满了
        # tyl：可以考虑最后一轮正常还是采样训练，多加一轮进行测试精度和测试水印（这样合理一点！）
        if iter == args.epochs:
            m = args.num_users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users.sort()
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        for ind, idx in enumerate(idxs_users):
            start_in = time.time()
            # tyl：这个if判断感觉没用啊，而且这个类对象不应该只需要每个client执行一次吗？
            if args.epochs == iter:
                # 都是500，没啥影响
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft],
                                    X=dict_X[idx], b=dict_b[idx])
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr],
                                    X=dict_X[idx], b=dict_b[idx])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            # tyl：确保w_local中head部分发生变化
            if args.alg != 'fedavg':
                for k in w_locals[idx].keys():
                    if k not in w_glob_keys:
                        w_local[k] = w_locals[idx][k]
            # 为什么还需要load_state_dict呢？我好像知道了
            net_local.load_state_dict(w_local)

            # tyl：这样逻辑就不太好了，训练和水印验证放在一起
            last = iter == args.epochs
            w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx,
                                                            w_glob_keys=w_glob_keys,
                                                            lr=args.lr, last=last, args=args, net_glob=net_glob)
            # zyb：分离水印嵌入和验证过程
            success_rate = local.validate(net=net_local.to(args.device))
            success_rates.append(success_rate)

            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]

            # zyb 保存head为numpy格式
            if not os.path.exists('./save/heads/'+str(args.frac)+'/'+str(args.embed_dim)):
                os.makedirs('./save/heads/'+str(args.frac)+'/'+str(args.embed_dim))
            np.save('./save/heads/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) +'_'+str(idx) +'_weight.npy',w_local['fc3.weight'].numpy())
            np.save('./save/heads/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) +'_'+str(idx) +'_bias.npy',w_local['fc3.bias'].numpy())
            # tyl：这里写的不太直观，解释一下
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key] * lens[idx]
                    # 为什么要这样赋值呢？为什么他不直接w_locals[idx] = w_local呢？
                    # 有一定道理的，确实确实，但也不是那么确实
                    w_locals[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += w_local[key] * lens[idx]
                    else:
                        w_glob[key] += w_local[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]

            # zyb:在最后一轮，所有人都验证所有人的水印，得出可以画热力图的数据
            if last and args.use_watermark:
                one_for_all_clients_rates = []
                for i in range(args.num_users):
                    one_for_all_clients_rates.append(local.validate(net=w_locals[i]))
                all_one_for_all_clients_rates.append(one_for_all_clients_rates)
                
            times_in.append(time.time() - start_in)
        # tyl:采样客户训练完毕   
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)

        if iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10:
            if not times:  # 等价于 if times == []
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                     w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
                                                     dataset_train=dataset_train, dict_users_train=dict_users_train,
                                                     return_all=False)
            accs.append(acc_test)
            # for algs which learn a single global model, these are the local accuracies (computed using the locally
            # updated versions of the global model at the end of each round)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    iter, loss_avg, loss_test, acc_test))
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model,
                # we fine-tune the head for 10 local epochs for fair comparison with FedRep
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    loss_avg, loss_test, acc_test))
            if iter >= args.epochs - 10 and iter != args.epochs:
                accs10 += acc_test / 10

            # below prints the global accuracy of the single global model for the relevant algs
            if args.alg == 'fedavg' or args.alg == 'prox':
                acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                         w_locals=None, indd=indd, dataset_train=dataset_train,
                                                         dict_users_train=dict_users_train, return_all=False)
                if iter != args.epochs:
                    print(
                        'Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                            iter, loss_avg, loss_test, acc_test))
                else:
                    print(
                        'Final Round, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                            loss_avg, loss_test, acc_test))
            if iter >= args.epochs - 10 and iter != args.epochs:
                accs10_glob += acc_test / 10

        if iter % args.save_every == args.save_every - 1:
            if not os.path.exists("./save/glob_models/" + str(args.frac)+'/'+str(args.embed_dim)):
                os.makedirs("./save/glob_models/" + str(args.frac)+'/'+str(args.embed_dim))
            model_save_path = "./save/glob_models/" + str(args.frac)+'/'+str(args.embed_dim)+'accs_' + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) + '_iter' + str(iter) + '.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end - start)
    print(times)
    print(accs)

    accs_dir = './save/accs_' + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
        args.shard_per_user) + str(args.use_watermark) +str(args.embed_dim)+str(args.frac) +  '.xlsx'
    if not os.path.exists(accs_dir):
        pd.DataFrame().to_excel(accs_dir)
    accs = np.array(accs)
    df = pd.read_excel(accs_dir)
    df['acc_seed{}'.format(rd)] = accs
    df.to_excel(accs_dir, index=False)

    succ_rates_dir = './save/success_rates' + args.alg + '_' + args.dataset + str(args.num_users) + '_' + str(
        args.shard_per_user) + str(args.use_watermark) +str(args.embed_dim)+str(args.frac) + '.xlsx'
    if not os.path.exists(succ_rates_dir):
        pd.DataFrame().to_excel(succ_rates_dir)
    success_rates = np.array(success_rates)
    df = pd.read_excel(succ_rates_dir)
    df['success_rates_seed{}'.format(rd)] = success_rates
    df.to_excel(succ_rates_dir, index=False)

    # 热力图的数据
    if args.use_watermark:
        all_detect_all_dir = './save/all_detect_all_rate' + args.alg + '_' + args.dataset + str(args.num_users) + '_' + str(
            args.shard_per_user) + str(args.use_watermark) + str(args.embed_dim)+str(args.frac) + '.xlsx'
        
        all_one_for_all_clients_rates = np.array(all_one_for_all_clients_rates)
        all_one_for_all_clients_rates = np.transpose(all_one_for_all_clients_rates)
        df = pd.DataFrame(all_one_for_all_clients_rates)
        df.to_excel(all_detect_all_dir)

if __name__ == '__main__':
    args = args_parser()
    # args.use_watermark = False
    # main(args=args)

    embed_dims = [64,128,192,256,320,384,448]
    fracs = [0.1,0.2,0.3]

    args.use_watermark = True
    for frac in fracs:
        args.frac = frac
        for embed_dim in embed_dims:
            args.embed_dim = embed_dim
            for i in range(10):
                main(args=args, rd=i, seed=i)
