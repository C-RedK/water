import numpy as np
import torch.nn as nn
from collections import OrderedDict

def get_X(net_glob, embed_dim, num_users, use_watermark=False):
    dict_X = {}
    p = net_glob.params()
    X_rows = p.numel()
    X_cols = embed_dim
    if not use_watermark:
        for i in range(num_users):
            dict_X[i] = None
        return dict_X

    for i in range(num_users):
        dict_X[i] = np.random.randn(X_rows, X_cols)
    return dict_X


def get_b(embed_dim, num_users, use_watermark=False):
    dict_b = {}
    if not use_watermark:
        for i in range(num_users):
            dict_b[i] = None
        return dict_b
    for i in range(num_users):
        dict_b[i] = np.random.randn(1, embed_dim)
        dict_b[i] = (np.sign(dict_b[i])+1) / 2
    return dict_b

#
def get_layer_weights_and_predict(model, w):
    if isinstance(model,nn.Module):
        p = model.params()
        p = p.cpu().view(1, -1).detach().numpy()
    elif isinstance(model,dict):
        # 反正是在最后一层嵌入，写死算求
        p = model['fc3.weight']
        p = p.cpu().view(1, -1).detach().numpy()
    pred_bparam = np.dot(p, w)  # dot product np.dot是矩阵乘法运算
    # print(pred_bparam)
    pred_bparam = np.sign(pred_bparam)
    pred_bparam = (pred_bparam + 1) / 2
    return pred_bparam

# 计算正确的比特数
def compute_BER(pred_b, b):
    correct_bit = np.logical_not(np.logical_xor(pred_b, b))
    correct_bit_num = correct_bit.sum()
    return correct_bit_num / pred_b.size
