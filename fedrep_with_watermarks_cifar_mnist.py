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
import os

from utils.options import args_parser
from utils.train_utils import get_model,getdata
from models.Update import LocalUpdate
from models.test import test_img_local_all
from utils.watermark import get_X, get_b

import time
import random

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)


def main(args,seed):
    # Step1:设置随机种子,保证结果可复现
    init_seed(seed=seed)
    # Step1：参数初始化
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # Step2：客户数据集初始化
    # lens 算是一个权重向量，每个元素代表每个客户的数据量,这里每个客户的数据量都是一样的
    lens = np.ones(args.num_users)
    # ck 修改get_data函数
    dataset_train, dataset_test, dict_users_train, dict_users_test = getdata(args)
    
    # Step3：模型初始化
    model_glob = get_model(args)
    model_glob.train() 
    # 如果有预训练模型，则加载预训练模型
    if args.load_fed != 'n': 
        fed_model_path = args.load_fed
        model_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(model_glob.state_dict().keys())
    print('net_glob.state_dict().keys():')
    print(model_glob.state_dict().keys())
    model_keys = [*model_glob.state_dict().keys()]

    # Step4：水印信息初始化
    # build watermark----------------------------------------------------------------------------------------------
    dict_X = get_X(net_glob=model_glob, embed_dim=args.embed_dim, num_users=args.num_users,use_watermark=args.use_watermark)
    dict_b = get_b(embed_dim=args.embed_dim, num_users=args.num_users,use_watermark=args.use_watermark)
    # build watermark----------------------------------------------------------------------------------------------

    # Step5：表示层参数与全局模型解耦
    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    layer_base = []
    if args.alg == 'fedrep' or args.alg == 'fedper':
        layer_base = [model_glob.weight_keys[i] for i in [0, 1, 3, 4]]
    if args.alg == 'fedavg':
        layer_base = []
        # 我他妈真醉了，真的服了,这里是把二维的变成一维的，然后后面的 in w_glob_keys才有用啊
    layer_base = list(itertools.chain.from_iterable(layer_base))

    print('total_num_layers:{}'.format(total_num_layers))
    print('w_glob_keys:{}'.format(layer_base)) #tyl：表示层模型keys
    print('net_keys:{}'.format(model_keys)) # tyl：全局模型keys
    print("learning rate:{}, batch size:{}".format(args.lr, args.local_bs))

    # Step6：Client模型参数初始化
    # 主要用来存放每个客户端的head参数
    # generate list of local models for each user
    model_clients = {}
    for user in range(args.num_users):
        model_local_dict = {}
        for key in model_glob.state_dict().keys():
            model_local_dict[key] = model_glob.state_dict()[key]
        model_clients[user] = model_local_dict 

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
        # 放全局参数的总和,加权计算使用
        state_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        # 最后一轮训练慢的原因是 最后一轮选择了全部客户端进行训练，所以才会这么慢
        # 原本可能选十几二十个，但是现在是所有的都他妈的要来训练，自然就满了
        # tyl：可以考虑最后一轮正常还是采样训练，多加一轮进行测试精度和测试水印（这样合理一点！）     
        #if iter == args.epochs:
        #    m = args.num_users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users.sort()

        times_in = []
        total_len = 0
        last = iter == args.epochs

        for ind, idx in enumerate(idxs_users):
            start_in = time.time()
            # tyl：这个if判断感觉没用啊，而且这个类对象不应该只需要每个client执行一次吗？
            if args.epochs == iter:
                # 都是500，没啥影响
                client = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft],
                                    X=dict_X[idx], b=dict_b[idx])
            else:
                client = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr],
                                    X=dict_X[idx], b=dict_b[idx])

            # 复制一份全局模型
            model_client = copy.deepcopy(model_glob)
            state_client = model_client.state_dict()
            #将上一轮的head参数更新到模型中
            if args.alg != 'fedavg':
                for k in model_clients[idx].keys():
                    if k not in layer_base:
                        state_client[k] = model_clients[idx][k]
            #上一轮的head和全局base 组成新的模型
            model_client.load_state_dict(state_client)

            # tyl：这样逻辑就不太好了，训练和水印验证放在一起
            state_client, loss, indd = client.train(net=model_client.to(args.device), idx=idx,
                                                            w_glob_keys=layer_base,
                                                            lr=args.lr, last=last, args=args, net_glob=model_glob)
            # zyb：分离水印嵌入和验证过程
            success_rate = client.validate(net=model_client.to(args.device))
            success_rates.append(success_rate)

            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]

            # zyb 保存head为numpy格式
            if not os.path.exists('./save/heads/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+str(args.epochs)):
                os.makedirs('./save/heads/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+str(args.epochs))
            np.save('./save/heads/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ str(args.epochs)+ '/' +args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) +'_'+str(idx) +'_weight.npy',state_client['fc3.weight'].cpu().numpy())
            np.save('./save/heads/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ str(args.epochs)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) +'_'+str(idx) +'_bias.npy',state_client['fc3.bias'].cpu().numpy())
            # tyl：这里写的不太直观，解释一下
            
            # 如果第一轮第一个client,直接赋值
            if len(state_glob) == 0:
                state_glob = copy.deepcopy(state_client)
                for k, key in enumerate(model_glob.state_dict().keys()):
                    state_glob[key] = state_glob[key] * lens[idx] # 这里是为了计算加权平均，先乘以每个client的数据量
                    model_clients[idx][key] = state_client[key]

            # 其他情况,参数直接相加
            else:
                for k, key in enumerate(model_glob.state_dict().keys()):
                    state_glob[key] += state_client[key] * lens[idx]        
                    model_clients[idx][key] = state_client[key]

            times_in.append(time.time() - start_in)

        # zyb:在最后一轮，所有人都验证所有人的水印，得出可以画热力图的数据
        if last and args.use_watermark:
            one_for_all_clients_rates = []
            for i in range(args.num_users):
                one_for_all_clients_rates.append(client.validate(net=model_clients[i]))
                all_one_for_all_clients_rates.append(one_for_all_clients_rates)             
        
        # tyl:采样客户训练完毕   
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in model_glob.state_dict().keys():
            state_glob[k] = torch.div(state_glob[k], total_len)
        
        # 好像没啥用
        state_client = model_glob.state_dict()
        for k in state_glob.keys():
            state_client[k] = state_glob[k]
        # 将全局模型更新为加权平均后的模型
        if args.epochs != iter:
            model_glob.load_state_dict(state_glob)

        if iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10:
            if not times:  # 等价于 if times == []
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            acc_test, loss_test = test_img_local_all(model_glob, args, dataset_test, dict_users_test,
                                                     w_glob_keys=layer_base, w_locals=model_clients, indd=indd,
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
                acc_test, loss_test = test_img_local_all(model_glob, args, dataset_test, dict_users_test,
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
            model_save_path = "./save/glob_models/" + str(args.frac)+'/'+str(args.embed_dim)+'/accs_' + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) + '_iter' + str(iter+1) + '.pt'
            torch.save(model_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end - start)
    print(times)
    print(accs)

    accs_dir = './save/accs/'
    accs_csv = './save/accs/accs_' + args.dataset + '_' + str(args.num_users) + '_' +  str(
        args.use_watermark)+'_' +str(args.embed_dim) + '_' + str(args.epochs)+'.csv'
    if not os.path.exists(accs_dir):
        os.makedirs(accs_dir)
    accs = np.array(accs)
    df = pd.DataFrame({'round': range(len(accs)), 'acc': accs})
    df.to_csv(accs_csv, index=False)
    
    # 热力图的数据
    if args.use_watermark:
        all_detect_all_dir = './save/water_acc/all_detect_all_rate' + args.alg + '_' + args.dataset + str(args.num_users) + '_' + str(
            args.shard_per_user) + str(args.use_watermark) + str(args.embed_dim)+str(args.frac) + '.xlsx'

        all_one_for_all_clients_rates = np.array(all_one_for_all_clients_rates)
        all_one_for_all_clients_rates = np.transpose(all_one_for_all_clients_rates)
        df = pd.DataFrame(all_one_for_all_clients_rates)
        df.to_excel(all_detect_all_dir)

if __name__ == '__main__':

    args = args_parser()
    embed_dims = [8,16,32,64,128,256,512]
    args.use_watermark = True
    for embed_dim in embed_dims:
        args.embed_dim = embed_dim
        main(args=args,seed=args.seed)
