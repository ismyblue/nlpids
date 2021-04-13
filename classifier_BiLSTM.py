# Name: classifier_BiLSTM.py
# Author: HuangHao
# Time: 2021/3/6 20:31

import os
import time
import tensorflow as tf
from tensorflow import keras

"""
分类器，可以通过此py文件获取一个分类器，用来分类流量
"""


class Classifier:
    """
    分类器，使用RNN网络，对文本进行分类
    """

    def __init__(self, vocab_size, weight_path=None):
        """
        构造函数
        :param vocab_size: 字典大小
        :param data:
        :param labels:
        :param test_size:
        """
        self.vocab_size = vocab_size
        self.weight_path = weight_path
        self.model = self.build_rnn_model()
        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                           metrics=['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),
                                    tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(),
                                    tf.keras.metrics.TrueNegatives(),
                                    tf.keras.metrics.FalseNegatives(), ])  # 训练过程中产生的损失和准确度
        # self.model.summary()
        # tf.keras.utils.plot_model(self.model, show_shapes=True, rankdir="LR")
        if self.weight_path is not None and os.path.exists(self.weight_path + '.index'):
            print('载入权重...')
            self.model.load_weights(self.weight_path)

    def build_rnn_model(self):
        """
        构建RNN，返回一个Model
        :return:
        """
        # 构建一个序列模型
        model = keras.Sequential()
        # 词嵌入层，word2vec
        model.add(keras.layers.Embedding(self.vocab_size, 16, mask_zero=True))
        # model.add(keras.layers.GlobalAveragePooling1D())
        # 双向LSTM层
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)))
        # 全连接层
        model.add(keras.layers.Dense(16, activation='relu'))
        # 输出
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        return model

    def train(self, data, labels, epochs, batch_size=128, validation_split=0.1, load_path=None, save_path=None):
        """
        模型训练
        :param epochs: 迭代次数
        :param batch_size: 批次大小
        :param load_path: 模型载入路径
        :param save_path: 模型保存路径
        :return:
        """

        # 如果传入了权重载入路径，那么载入权重
        if load_path is not None and os.path.exists(load_path):
            print('载入权重文件{}'.format(load_path))
            self.model.load_weights(load_path)
        # 如果传入了权重保存路径，那么创建一个模型保存回调函数
        if save_path is not None:
            # 创建一个保存模型权重的回调
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True,
                                                             save_freq='epoch', verbose=1)
            history_callback = tf.keras.callbacks.History()
            history = self.model.fit(data, labels, validation_split=validation_split, batch_size=batch_size,
                                     epochs=epochs, callbacks=[cp_callback])
        else:
            history = self.model.fit(data, labels, validation_split=validation_split, batch_size=batch_size,
                                     epochs=epochs)
        return history


def show_history(history_dict):
    """
    输出数据体，训练过程中的loss, acc, recall, precsion
    :param history:
    :return:
    """
    loss = history_dict['loss']
    acc = history_dict['acc']
    recall = history_dict['recall']
    precision = history_dict['precision']
    val_loss = history_dict['val_loss']
    val_acc = history_dict['val_acc']
    val_recall = history_dict['val_recall']
    val_precision = history_dict['val_precision']
    plt.figure()
    plt.plot(np.arange(0, len(loss)), loss, label='loss')
    plt.plot(np.arange(0, len(val_loss)), val_loss, label='val_loss')
    plt.legend()

    plt.figure()
    plt.plot(np.arange(0, len(acc)), acc, label='acc')
    plt.plot(np.arange(0, len(recall)), recall, label='recall')
    plt.plot(np.arange(0, len(precision)), precision, label='precision')
    plt.plot(np.arange(0, len(val_acc)), val_acc, label='val_acc')
    plt.plot(np.arange(0, len(val_recall)), val_recall, label='val_recall')
    plt.plot(np.arange(0, len(val_precision)), val_precision, label='val_precision')
    plt.legend()
    plt.show()


http_text0 = """GET http://localhost:8080/tienda1/imagenes/3.gif HTTP/1.1
User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
Pragma: no-cache
Cache-control: no-cache
Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5
Accept-Encoding: x-gzip, x-deflate, gzip, deflate
Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5
Accept-Language: en
Host: localhost:8080
Cookie: JSESSIONID=4567793E184E0925234DADCEECD6999A
Connection: close"""

http_text1 = """GET http://localhost:8080/tienda1/imagenes/nuestratierra.jpg HTTP/1.1
User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
Pragma: no-cache
Cache-control: no-cache
Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5
Accept-Encoding: x-gzip, x-deflate, gzip, deflate
Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5
Accept-Language: en
Host: localhost:8080
Cookie: JSESSIONID=A70DD1BA160B294CB5E1C2D8FAE7C09F
Connection: close"""

http_text2 = """GET http://localhost:8080/tienda1/imagenes/nuestratierra.jpg.BAK HTTP/1.1
User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
Pragma: no-cache
Cache-control: no-cache
Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5
Accept-Encoding: x-gzip, x-deflate, gzip, deflate
Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5
Accept-Language: en
Host: localhost:8080
Cookie: JSESSIONID=3CC12010CDA952F123240EBAD79B55CC
Connection: close"""

http_text3 = """GET http://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=Jam%F3n+Ib%E9rico&precio=85&cantidad=%27%3B+DROP+TABLE+usuarios%3B+SELECT+*+FROM+datos+WHERE+nombre+LIKE+%27%25&B1=A%F1adir+al+carrito HTTP/1.1
User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
Pragma: no-cache
Cache-control: no-cache
Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5
Accept-Encoding: x-gzip, x-deflate, gzip, deflate
Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5
Accept-Language: en
Host: localhost:8080
Cookie: JSESSIONID=B92A8B48B9008CD29F622A994E0F650D
Connection: close"""

from dataset import load_http_dataset_csic_2010, load_http_text_list
from keywords import keywords_dict_size, get_keywords_dict
from sklearn.model_selection import train_test_split
from preprocessor import PreProcessor
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random

if __name__ == '__main__':
    # 获取数据和标签
    (x_train, y_train), _, (x_test, y_test) = load_http_dataset_csic_2010()
    print("字典大小：", keywords_dict_size())
    # 构造分类器
    cls = Classifier(keywords_dict_size(), weight_path='training/cp.ckpt')
    cls.model.summary()
    # exit(0)
    # # 训练
    history = cls.train(x_train, y_train, validation_split=0.1, epochs=100, batch_size=128,
                        load_path='training/cp.ckpt', save_path='training/cp.ckpt')
    pickle.dump(history.history, open('history{}.pickle'.format(time.time()), 'wb'))
    # 评估
    cls.model.evaluate(x_test, y_test, verbose=1, batch_size=128)

    # 预测
    http_text_list = [http_text0, http_text1, http_text2, http_text3]
    pre = PreProcessor(get_keywords_dict())
    test_data = []
    for http_text in http_text_list:
        digital = pre.format_unification_digital_list(http_text)
        test_data.append(digital)
        print(len(digital), digital)
        print(pre.format_unification_text(http_text))
    # 填充定长
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=x_train.shape[1], padding='post')

    pre_labels = cls.model.predict(test_data, verbose=1)
    for label in pre_labels:
        if label >= 0.5:
            print('异常', end=' ')
        else:
            print('正常', end=' ')

    # history_dict = pickle.load(open('history{}.pickle'.format(random.random()), 'rb'))
    # show_history(history_dict)
