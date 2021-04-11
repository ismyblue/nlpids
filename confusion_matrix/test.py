import pickle
import numpy as np
import matplotlib.pyplot as plt

matrix = pickle.load(open('None_fit.pkl', 'rb'))
matrix = np.array(matrix)


def calc_binary_metrics(matrix):
    """
    计算二分类的评价指标，
    :param matrix: 混淆矩阵列表
    :return:
    """
    TN = matrix[:, 0, 0]
    FN = matrix[:, 0, 1]
    FP = matrix[:, 1, 0]
    TP = matrix[:, 1, 1]
    acc = (TP + TN) / (TP + TN + FP + FN)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return acc, TPR, FPR


acc, TPR, FPR = calc_binary_metrics(matrix)
plt.figure()
plt.plot(np.arange(0, len(acc)), acc, label='acc')
plt.plot(np.arange(0, len(TPR)), TPR, label='TPR')
plt.legend()
plt.show()



