import copy
import itertools
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from utils.train_utils import  get_model,getdata
from utils.options import args_parser
from utils.watermark import get_X, get_b
from torch.utils.data import DataLoader, Dataset
from utils.watermark import get_layer_weights_and_predict,compute_BER
from models.test import test_img_local_all


def pruning_resnet(model, pruning_perc):

    if pruning_perc == 0:
        return
    allweights = []
    for p in model.parameters():
        allweights += p.data.cpu().abs().numpy().flatten().tolist()
    
    allweights = np.array(allweights)
    threshold = np.percentile(allweights, pruning_perc)
    for p in model.parameters():
        mask = p.abs() > threshold
        p.data.mul_(mask.float())

def prune_linear_layer(layer, pruning_method, amount):
    """
    对给定的linear层进行剪枝
    :param layer: 待剪枝的linear层
    :param pruning_method: 剪枝方法
    :param amount: 剪枝比例
    """
    # 按照指定的方法进行剪枝
    pruning_method.apply(layer, 'weight', amount)
    # 剪枝永久生效
    prune.remove(layer, 'weight')

  

# 计算水印准确度
def validate(net,X,b):
    X = torch.tensor(X, dtype=torch.float32).to(args.device)
    b = torch.tensor(b, dtype=torch.float32).to(args.device)
    success_rate = -1
    pred_b = get_layer_weights_and_predict(net,X.cpu().numpy())
    success_rate = compute_BER(pred_b=pred_b,b=b.cpu().numpy())
    return success_rate

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)

def main(args,loadpath):
    # 随机种子
    init_seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # 加载数据集,这是使用所有的测试集进行测试
    dataset_train, dataset_test, dict_users_train, dict_users_test = getdata(args)
    # 获取模型
    model = get_model(args)
    # build watermark----------------------------------------------------------------------------------------------
    dict_X = get_X(net_glob=model, embed_dim=args.embed_dim, num_users=args.num_users,use_watermark=args.use_watermark)
    dict_b = get_b(embed_dim=args.embed_dim, num_users=args.num_users,use_watermark=args.use_watermark)
    # build watermark----------------------------------------------------------------------------------------------
    
    # 加载要测试的模型参数
    sd = torch.load(loadpath)
    prunedf = []
    res = {}

    for perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 82,84,86,88,90]:
        # 加载模型
        model.load_state_dict(sd)
        w_locals = {}
        for i in range(args.num_users):
            w_local_dict = {}
            for key in model.state_dict().keys():
               w_local_dict[key] = model.state_dict()[key]
            # 加载fc3的参数 转换为torch.tensor
            w_local_dict['fc3.weight'] = torch.from_numpy(np.load('./save/heads/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) +'_'+str(i) +'_'+str(args.epochs)+'_weight.npy'))
            w_local_dict['fc3.bias'] = torch.from_numpy(np.load('./save/heads/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) +'_'+str(i) +'_'+str(args.epochs)+'_bias.npy'))

            w_locals[i] = w_local_dict
        
        #对每个clinet进行剪枝
        for i in range(args.num_users):
            pruned_model = copy.deepcopy(model)
            pruned_model.load_state_dict(w_locals[i])
            #pruning_resnet(pruned_model, perc)
            amount = perc/100
            #prune.random_unstructured(pruned_model.fc3, name="weight", amount=amount)
            prune.l1_unstructured(pruned_model.fc3, name="weight", amount=amount)
            #prune.ln_structured(pruned_model.fc3, name="weight", amount=amount, n=2, dim=0)
            prune.remove(pruned_model.fc3, 'weight')
            w_locals[i] = pruned_model.state_dict()
      
        #剪枝后模型的性能
        indd = None # 特定数据集的参数
        # 表示层
        #w_glob_keys = [model.weight_keys[i] for i in [0, 1, 3, 4]]
        #w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
        
        #这里计算的是对每个client的平均acc loss
        #不加w_glob_keys的话，会计算所有的参数的acc loss 
        acc_test, loss_test = test_img_local_all(model, args, dataset_test, dict_users_test,
                                                 w_locals=w_locals,indd=indd,
                                                 dataset_train=dataset_train, dict_users_train=dict_users_train,
                                                 return_all=False)
        #剪枝后水印检测准确率: 对每个client的水印进行检测,计算一个平均值
        #考虑到，有的client可能一次也没有参与过训练，model中不存在水印，会影响精确度
        all_success_rate = []
        for i in range(args.num_users):
         # model.load_state_dict(w_locals[i])
          w_model = copy.deepcopy(model)
          w_model.load_state_dict(w_locals[i])
          success_rate = validate(net=w_model.to(args.device),X=dict_X[i],b=dict_b[i])
          all_success_rate.append(success_rate)
          #print(success_rate)

        acc_watermark = sum(all_success_rate) / args.num_users

        res['perc'] = perc
        res['acc_watermark'] = acc_watermark
        res['acc_model']     = acc_test
        print('prec: {:3d}, acc_watermark: {: 3f}, acc_model: {: 3f}'.format(perc,acc_watermark,acc_test))
        prunedf.append(res)
    # 保存到csv文件,文件名为：模型名+数据集名+用户数+每个用户的数据量+epoch+剪枝率
    #如果没有这个文件，就创建一个
    if not os.path.exists('./save/prunedf/'+str(args.frac)+'/'+str(args.embed_dim)):
        os.makedirs('./save/prunedf/'+str(args.frac)+'/'+str(args.embed_dim))
    pd.DataFrame(prunedf).to_csv('./save/prunedf/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) +'_'+str(args.epochs)+'_'+str(args.perc)+'.csv')
    
if __name__ == '__main__':

   args = args_parser()
   emds = [8,16,32,64,128,256,512]
   for emd in emds:
        args.embed_dim = emd
        args.epoch = 50
        main(args,loadpath='./save/model/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) +'_'+str(args.epochs)+'.pt')
        args.epoch = 100
        main(args,loadpath='./save/model/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) +'_'+str(args.epochs)+'.pt')

