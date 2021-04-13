import pickle
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt


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
    precision = (TP) / (TP + FP)
    TPR = TP / (TP + FN)
    FPR = FP / (TP + TN)
    return acc, precision, TPR, FPR


acc_dict = {}
precision_dict = {}
TPR_dict = {}
FPR_dict = {}
for file in os.listdir():
    if 'pkl' not in file:
        continue
    obj = pickle.load(open(file, 'rb'))
    matrix = np.array(obj['confusion_matrix_list'])
    predict_list = np.array(obj['predict_list'])
    y_pre = tf.argmax(predict_list[-1], 1)
    if 'evaluate' in file:
        print(file)
        print(matrix)
        continue
    acc, precision, TPR, FPR = calc_binary_metrics(matrix)
    acc_dict[file] = acc
    precision_dict[file] = precision
    TPR_dict[file] = TPR
    FPR_dict[file] = FPR

# 画ACC
plt.figure()
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Percentage')
for file_name in acc_dict:
    acc = acc_dict[file_name]
    plt.plot(np.arange(0, len(acc)), acc, label=file_name.split('.')[0])
plt.legend()

# # 画precision
# plt.figure()
# plt.title('Precision')
# plt.xlabel('Epochs')
# plt.ylabel('Percentage')
# for file_name in TPR_dict:
#     precision = precision_dict[file_name]
#     plt.plot(np.arange(0, len(precision)), precision, label=file_name.split('.')[0])
# plt.legend()

# 画TPR
plt.figure()
plt.title('TPR')
plt.xlabel('Epochs')
plt.ylabel('Percentage')
for file_name in TPR_dict:
    TPR = TPR_dict[file_name]
    plt.plot(np.arange(0, len(TPR)), TPR, label=file_name.split('.')[0])
plt.legend()


# # 画FPR
# plt.figure()
# plt.title('FPR')
# plt.xlabel('Epochs')
# plt.ylabel('Percentage')
# for file_name in TPR_dict:
#     FPR = FPR_dict[file_name]
#     plt.plot(np.arange(0, len(FPR)), FPR, label=file_name.split('.')[0])
# plt.legend()

plt.show()



