"""
重采样技术：过采样、欠采样、两者结合、...
"""

import pickle
from dataset import *
from keywords import keywords_dict_size
from classifier_multihead import WebAttackClassifier, CustomSchedule

from tensorflow import keras
from tensorflow.keras.metrics import Metric

from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek


def balance_data(method, x_train, y_train):
    """
    平衡样本
    :param method:
    :param x_train:
    :param y_train:
    :return: x_train, y_train
    """
    """    
    # 过采样：RandomOverSampler/SMOTE/ADASYN/BorderlineSMOTE
    # 欠采样：RandomUnderSampler/ClusterCentroids/NearMiss/TomekLinks
    # 两者结合：SMOTETomek/SMOTEENN
    """
    if method == 'None':
        return x_train, y_train
    elif method == 'RandomOverSampler':
        resamper = RandomOverSampler(random_state=0)
        return resamper.fit_resample(x_train, y_train)
    elif method == 'SMOTE':
        resamper = SMOTE(random_state=0)
        return resamper.fit_resample(x_train, y_train)
    elif method == 'ADASYN':
        resamper = ADASYN(random_state=0)
        return resamper.fit_resample(x_train, y_train)
    elif method == 'RandomUnderSampler':
        resamper = RandomUnderSampler(random_state=0)
        return resamper.fit_resample(x_train, y_train)
    elif method == 'BorderlineSMOTE':
        resamper = BorderlineSMOTE(random_state=0)
        return resamper.fit_resample(x_train, y_train)
    elif method == 'ClusterCentroids':
        resamper = ClusterCentroids(random_state=0)
        return resamper.fit_resample(x_train, y_train)
    elif method == 'NearMiss':
        resamper = NearMiss()
        return resamper.fit_resample(x_train, y_train)
    elif method == 'TomekLinks':
        resamper = TomekLinks()
        return resamper.fit_resample(x_train, y_train)
    elif method == 'SMOTETomek':
        resamper = SMOTETomek(random_state=0)
        return resamper.fit_resample(x_train, y_train)
    elif method == 'SMOTEENN':
        resamper = SMOTEENN(random_state=0)
        return resamper.fit_resample(x_train, y_train)


class Confusion_Matrix_Saver(keras.callbacks.Callback):
    """
    自定义回调函数，用于保存每epoch的混淆矩阵
    """

    def __init__(self, filepath, num_classes, x, y):
        """
        混淆矩阵保存回调
        :param filepath: 
        :param num_classes: 
        :param x: 训练数据
        :param y: 训练标签
        """
        self.filepath = filepath
        self.num_classes = num_classes
        self.x = x
        self.y = tf.argmax(y, axis=1)

    def on_train_begin(self, logs):
        self.confusion_matrix_list = []

    def on_epoch_end(self, epoch, logs):
        y_pre = self.model.predict(self.x)
        y_pre = tf.argmax(y_pre, axis=1)
        matrix = np.zeros(shape=(self.num_classes, self.num_classes), dtype='int32')
        for i in range(y_pre.shape[0]):
            matrix[self.y[i]][y_pre[i]] += 1
        self.confusion_matrix_list.append(matrix)
        # 每一epoch结束，保存一下所有的混淆矩阵
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.confusion_matrix_list, f)
        print('第{}个epoch的混淆矩阵保存成功...'.format(epoch))

    def on_test_begin(self, logs):
        self.confusion_matrix_list = []

    def on_test_end(self, logs):
        y_pre = self.model.predict(self.x)
        y_pre = tf.argmax(y_pre, axis=1)
        matrix = np.zeros(shape=(self.num_classes, self.num_classes), dtype='int32')
        for i in range(y_pre.shape[0]):
            matrix[self.y[i]][y_pre[i]] += 1
        self.confusion_matrix_list.append(matrix)
        # 每一epoch结束，保存一下所有的混淆矩阵
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.confusion_matrix_list, f)
        print('evaluate的混淆矩阵保存成功...')


if __name__ == '__main__':
    """
    样本平衡选择实验
    """
    # 获取字典大小
    vocab_size = keywords_dict_size()
    print("vocab_size:", vocab_size)

    resampling_methods = ['None', 'RandomOverSampler', 'SMOTE', 'ADASYN', 'BorderlineSMOTE', 'RandomUnderSampler',
                          'ClusterCentroids', 'NearMiss', 'TomekLinks', 'SMOTETomek', 'SMOTEENN']
    # 选用不同的方法进行样本平衡
    for i in range(0, len(resampling_methods)):
        method = resampling_methods[i]
        # 获取数据
        (x_train, y_train), (x_test, y_test) = load_http_dataset_csic_2010()
        # x_train = x_train[:200]
        # y_train = y_train[:200]
        # x_test = x_test[:200]
        # y_test = y_test[:200]
        print('样本平衡前: y_train', sorted(Counter(y_train).items()))

        # 样本平衡
        print('使用{}方法进行样本平衡'.format(method))
        x_train, y_train = balance_data(method, x_train, y_train)
        print('样本平衡后：y_train', sorted(Counter(y_train).items()))

        # 保存平衡后的样本
        print('保存{}方法平衡后的样本..'.format(method))
        with open('resampling_data/{}.pkl'.format(method), 'wb') as f:
            pickle.dump({'x_train': x_train, 'y_train': y_train}, f)

        # 构造分类器
        webAttackClassifier = WebAttackClassifier(num_layers=1, d_model=16, num_heads=8, dff=16,
                                                  input_vocab_size=vocab_size, categories=2)
        # 学习速率
        learning_rate = CustomSchedule(16)
        # 优化器
        optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # 指定模型优化器、loss、指标
        webAttackClassifier.compile(optimizer=optimizer,
                                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                                    metrics=[tf.keras.metrics.CategoricalAccuracy(),
                                             tf.keras.metrics.Precision(),
                                             tf.keras.metrics.Recall(),
                                             ])

        # 训练##############################################
        print('开始训练...')
        # one_hot 标签
        y_train = tf.squeeze(tf.one_hot(y_train, 2))

        # history记录回调
        csv_logger = tf.keras.callbacks.CSVLogger('history/{}_fit.csv'.format(method))
        # 保存混淆矩阵的回调
        confusion_matrix_saver = Confusion_Matrix_Saver('confusion_matrix/{}_fit.pkl'.format(method), num_classes=2,
                                                        x=x_train, y=y_train)
        # 保存模型的回调
        model_saver = keras.callbacks.ModelCheckpoint(filepath="checkpoints/{}_multihead".format(method),
                                                      save_freq='epoch', verbose=1)
        # 开始训练
        webAttackClassifier.fit(x_train, y_train, batch_size=64, epochs=50,
                                callbacks=[csv_logger, confusion_matrix_saver, model_saver])

        # 评估##############################################
        print('开始评估...')
        y_test = tf.squeeze(tf.one_hot(y_test, 2))
        # 混淆矩阵保存
        confusion_matrix_saver = Confusion_Matrix_Saver('confusion_matrix/{}_evaluate.pkl'.format(method),
                                                        num_classes=2,
                                                        x=x_test, y=y_test)
        # 评估
        fit_history = webAttackClassifier.evaluate(x_test, y_test, batch_size=64, callbacks=[confusion_matrix_saver])

        # 删除空间，避免OOM
        del x_train
        del y_train
        del x_test
        del y_test
        del confusion_matrix_saver
        del webAttackClassifier
