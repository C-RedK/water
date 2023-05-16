import torch
import torch.nn as nn

def get_X(net_glob, embed_dim, num_users, use_watermark=False):
    dict_X = {}
    p = net_glob.head_params()
    X_rows = 0
    for i in p:
        X_rows += i.numel()
    X_cols = embed_dim
    if not use_watermark:
        for i in range(num_users):
            dict_X[i] = None
        return dict_X

    for i in range(num_users):
        dict_X[i] = torch.randn(X_rows, X_cols)
    return dict_X


def get_b(embed_dim, num_users, use_watermark=False):
    dict_b = {}
    if not use_watermark:
        for i in range(num_users):
            dict_b[i] = None
        return dict_b
    for i in range(num_users):
        dict_b[i] = torch.randn(1, embed_dim)
        dict_b[i] = (torch.sign(dict_b[i])+1) / 2
    return dict_b

#
def get_layer_weights_and_predict(model, x,device):
    if isinstance(model,nn.Module):
        #p = model.head_params()
        #p = p.cpu().view(1, -1).detach().numpy()
        # 得到参数 转化为一维向量
        p = model.head_params()
        # y 是一个一维向量，用来存储参数
        y = torch.tensor([], dtype=torch.float32).to(device)
        #将每个张量转化为一维向量，然后拼接在一起
        for i in p:
            y = torch.cat((y, i.view(-1)), 0)
        y = y.view(1,-1).to(device)

    elif isinstance(model,dict):
        #抛出异常
        raise Exception("model is a dict")
    pred_bparam = torch.matmul(y, x.to(device))  # dot product np.dot是矩阵乘法运算
    # print(pred_bparam)
    pred_bparam = torch.sign(pred_bparam.to(device))
    pred_bparam = (pred_bparam + 1) / 2
    return pred_bparam

# 计算正确的比特数
def compute_BER(pred_b, b, device):
    correct_bit = torch.logical_not(torch.logical_xor(pred_b, b))
    correct_bit_num = torch.sum(correct_bit)
    #print(correct_bit_num,pred_b.size(),pred_b.size(0))
    res = correct_bit_num / pred_b.size(1)
    return res.item()
