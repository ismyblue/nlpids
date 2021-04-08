# Name: nlpids.py
# Author: HuangHao
# Time: 2021/3/6 20:38



http_text = """GET http://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=Jam%F3n+Ib%E9rico&precio=85&cantidad=%27%3B+DROP+TABLE+usuarios%3B+SELECT+*+FROM+datos+WHERE+nombre+LIKE+%27%25&B1=A%F1adir+al+carrito HTTP/1.1
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

from sklearn.utils import shuffle
# from dataset import calc_high_freq_words
from tensorflow import keras
from keywords import get_keywords_dict, keywords_dict_size
from preprocessor import PreProcessor
from classifier_BiLSTM import Classifier
from dataset import load_http_text_list, split_dataset
import pickle
from dataset import load_http_dataset_csic_2010
from balancer_gan import Balancer
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # pre = PreProcessor(get_keywords_dict())

    history_dict = pickle.load(open('history1.pickle', 'rb'))

    for key in history_dict:
        print(key, history_dict[key][-1])

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





