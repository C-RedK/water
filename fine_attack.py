import copy
import itertools
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
            w_local_dict['fc3.weight'] = torch.from_numpy(np.load('./save_fine/heads/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) +'_'+str(i) +'_weight.npy'))
            w_local_dict['fc3.bias'] = torch.from_numpy(np.load('./save_fine/heads/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) +'_'+str(i) +'_bias.npy'))

            w_locals[i] = w_local_dict
        
 
        indd = None # 特定数据集的参数
        
        # 模型测试
        #这里计算的是对每个client的平均acc loss
        #不加w_glob_keys的话，会计算所有的参数的acc loss 
        acc_test, loss_test = test_img_local_all(model, args, dataset_test, dict_users_test,
                                                 w_locals=w_locals,indd=indd,
                                                 dataset_train=dataset_train, dict_users_train=dict_users_train,
                                                 return_all=False)
        # 水印检测
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

'''
    dirname = f'logs/pruning_attack/{loadpath.split("/")[1]}/{loadpath.split("/")[2]}'
    os.makedirs(dirname, exist_ok=True)
    histdf = pd.DataFrame(prunedf)
    histdf.to_csv(f'{dirname}/history-{args.dataset}.csv')
'''
if __name__ == '__main__':

    args = args_parser()
    args.embed_dim = 64
    main(args,loadpath="")