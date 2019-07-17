import numpy as np

def mae(y_true, y_pred):
    diff = y_true - y_pred
    diff_abs = np.abs(diff)
    soma = diff_abs.sum()
    media_soma = soma / y_true.shape[0]
    return media_soma

def msle(y_true, y_pred):
    return ((np.log(y_true+1) - np.log(y_pred+1))**2).sum() / y_true.shape[0]
