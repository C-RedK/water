import numpy as np


def get_X(net_glob, embed_dim, num_users):
    dict_X = {}
    p = net_glob.params()
    X_rows = p.numel()
    X_cols = embed_dim
    for i in range(num_users):
        dict_X[i] = np.random.randn(X_rows, X_cols)

    return dict_X


def get_b(embed_dim, num_users):
    dict_b = {}
    for i in range(num_users):
        dict_b[i] = np.random.randn(1, embed_dim)
    return dict_b


def get_layer_weights_and_predict(model, w):
    p = model.param()
    p = p.cpu().view(1, -1).detach().numpy()
    pred_bparam = np.dot(p, w)  # dot product np.dot是矩阵乘法运算
    # print(pred_bparam)
    pred_bparam = np.sign(pred_bparam)
    pred_bparam = (pred_bparam + 1) / 2
    return pred_bparam


def compute_BER(pred_b, b):
    correct_bit = np.logical_and(pred_b, b)
    correct_bit_num = correct_bit.sum()
    return correct_bit_num / pred_b.size
