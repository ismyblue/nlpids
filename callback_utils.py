import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle


class Confusion_Matrix_Saver(keras.callbacks.Callback):
    """
    自定义回调函数，用于保存每epoch的混淆矩阵
    """

    def __init__(self, filepath, num_classes, x, y):
        """
        混淆矩阵保存回调
        :param filepath:
        :param num_classes:
        :param x: 数据
        :param y: 标签
        """
        self.filepath = filepath
        self.num_classes = num_classes
        self.x = x
        self.y = tf.argmax(y, axis=1)

    def on_train_begin(self, logs):
        # 混淆矩阵，用来计算acc, TPR
        self.confusion_matrix_list = []
        # 用来画ROC曲线
        self.predict_list = []

    def on_epoch_end(self, epoch, logs):
        y_pre = self.model.predict(self.x)
        # 添加预测结果
        self.predict_list.append(y_pre)

        y_pre = tf.argmax(y_pre, axis=1)
        matrix = np.zeros(shape=(self.num_classes, self.num_classes), dtype='int32')
        for i in range(y_pre.shape[0]):
            matrix[self.y[i]][y_pre[i]] += 1
        # 添加混淆矩阵
        self.confusion_matrix_list.append(matrix)

        # 每一epoch结束，保存一下所有的混淆矩阵和预测结果
        with open(self.filepath, 'wb') as f:
            pickle.dump({'confusion_matrix_list': self.confusion_matrix_list, 'predict_list': self.predict_list}, f)
        print('第{}个epoch的混淆矩阵和预测结果保存成功...'.format(epoch))

    def on_test_begin(self, logs):
        # 混淆矩阵，用来计算acc, TPR
        self.confusion_matrix_list = []
        # 用来画ROC曲线
        self.predict_list = []

    def on_test_end(self, logs):
        y_pre = self.model.predict(self.x)
        # 添加预测结果
        self.predict_list.append(y_pre)

        y_pre = tf.argmax(y_pre, axis=1)
        matrix = np.zeros(shape=(self.num_classes, self.num_classes), dtype='int32')
        for i in range(y_pre.shape[0]):
            matrix[self.y[i]][y_pre[i]] += 1
        # 添加混淆矩阵
        self.confusion_matrix_list.append(matrix)

        # 每一epoch结束，保存一下所有的混淆矩阵和预测结果
        with open(self.filepath, 'wb') as f:
            pickle.dump({'confusion_matrix_list': self.confusion_matrix_list, 'predict_list': self.predict_list}, f)
        print('混淆矩阵和预测结果保存成功...')
