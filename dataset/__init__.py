# Name: __init__.py
# Author: HuangHao
# Time: 2021/3/8 19:33

import tensorflow as tf
from preprocessor import PreProcessor, get_keywords_dict
import pickle
import os
import numpy as np
from sklearn.utils import shuffle


def load_http_text_list(traffic_txt_file):
    """
    载入http_text文本数据，
    :return: http_text_list
    """
    http_text_list = []
    with open(traffic_txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        http_text = ''
        # 回车次数
        renturn_count = 0
        for line in lines:
            if line == '\n':
                renturn_count += 1
                # 如果回车次数等于两个，说明一个样本结束
                if renturn_count == 2:
                    http_text_list.append(http_text)
                    renturn_count = 0
                    http_text = ''
                else:
                    http_text += line
            else:
                http_text += line
    return http_text_list


def load_http_dataset_csic_2010():
    """
    载入http dataset csis 2010数据集，返回一个tf.data.Dataset
    :return: (x_train, y_train), (x_validate, y_validate), (x_test, y_test)
    """
    if os.path.exists('dataset/csic_2010_dataset.pkl'):
        csic_2010_dataset = pickle.load(open('dataset/csic_2010_dataset.pkl', 'rb'))
        return csic_2010_dataset
    else:
        return generate_http_dataset_csic_2010()
        # csic_2010_dataset = pickle.load(open('dataset/csic_2010_dataset.pkl', 'rb'))
        # return csic_2010_dataset


def generate_http_dataset_csic_2010():
    """
    预处理http dataset csis 2010数据集，生成一个tf.data.Dataset，保存
    :return: (x_train, y_train), (x_validate, y_validate), (x_test, y_test)
    """
    # 获取预处理后的样本list
    # 如果已经有序列化好的list对象，那么直接读取
    if os.path.exists('dataset/x_train.pkl') \
            and os.path.exists('dataset/y_train.pkl') \
            and os.path.exists('dataset/x_validate.pkl') \
            and os.path.exists('dataset/y_validate.pkl') \
            and os.path.exists('dataset/x_test.pkl') \
            and os.path.exists('dataset/y_test.pkl'):
        # 读取序列化对象
        x_train = pickle.load(open('dataset/x_train.pkl', 'rb'))
        y_train = pickle.load(open('dataset/y_train.pkl', 'rb'))
        x_validate = pickle.load(open('dataset/x_validate.pkl', 'rb'))
        y_validate = pickle.load(open('dataset/y_validate.pkl', 'rb'))
        x_test = pickle.load(open('dataset/x_test.pkl', 'rb'))
        y_test = pickle.load(open('dataset/y_test.pkl', 'rb'))
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_validate shape:", x_validate.shape)
        print("y_validate shape:", y_validate.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)
        return (x_train, y_train), (x_validate, y_validate), (x_test, y_test)
    # 如果没有序列化好的list对象，那么就预处理然后生成
    else:
        # 创建一个预处理器
        pre = PreProcessor(get_keywords_dict())
        # 生成并保存异常样本 label=0
        anomalous_unified_format_list = []
        # 获取http文本列表
        http_text_list = load_http_text_list('dataset/anomalousTrafficTest.txt')
        # 进行预处理，生成统一格式数据
        for http_text in http_text_list:
            anomalous_unified_format_list.append(pre.format_unification_digital_list(http_text))
        pickle.dump(anomalous_unified_format_list, open('dataset/anomalousUnifiedFormat.pkl', 'wb'))

        # 生成并保存正常样本 label=1
        normal_unified_format_list = []
        # 获取http文本列表
        http_text_list = load_http_text_list('dataset/normalTrafficTest.txt')
        http_text_list += load_http_text_list('dataset/normalTrafficTraining.txt')
        # 进行预处理，生成统一格式数据
        for http_text in http_text_list:
            normal_unified_format_list.append(pre.format_unification_digital_list(http_text))
        pickle.dump(normal_unified_format_list, open('dataset/normalUnifiedFormat.pkl', 'wb'))

    # 样本数据
    data = np.append(np.array(anomalous_unified_format_list, dtype=object),
                     np.array(normal_unified_format_list, dtype=object))

    # 可变长度填充为定长
    data = tf.keras.preprocessing.sequence.pad_sequences(data, padding="post")

    # 标签，1为异常anomalous(阳性), 0为正常normal(阴性)
    # labels = np.append(np.ones(len(anomalous_unified_format_list), dtype=np.int32),
    #                    np.zeros(len(normal_unified_format_list), dtype=np.int32))
    # print("数据集大小:{} 标签集大小:{}".format(data.shape, labels.shape))

    # 切分数据集,洗牌，并保存
    (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = split_dataset(
        data[:len(anomalous_unified_format_list)], data[len(anomalous_unified_format_list):], validate_size=0.2,
        test_size=0.2)
    return (x_train, y_train), (x_validate, y_validate), (x_test, y_test)


def split_dataset(anomalous_unified_format_data, normal_unified_format_data, validate_size=0.2, test_size=0.2):
    """
    划分数据集
    :param anomalous_unified_format_data: 填充了长度的格式统一的异常样本
    :param normal_unified_format_data: 填充了长度的格式统一的正常样本
    :param validate_size: 验证集比例
    :param test_size: 测试集比例
    :return:
    """
    # 洗牌
    anomalous_unified_format_data = shuffle(anomalous_unified_format_data, random_state=100)
    # 洗牌
    normal_unified_format_data = shuffle(normal_unified_format_data, random_state=100)
    print("异常样本总共：", len(anomalous_unified_format_data))
    print("正常样本总共：", len(normal_unified_format_data))

    # 训练集
    # 训练集(正常/异常)样本个数
    a_train_count = int(len(anomalous_unified_format_data) * (1 - test_size - validate_size))
    n_train_count = int(len(normal_unified_format_data) * (1 - test_size - validate_size))
    # 验证集(正常/异常)样本个数
    a_validate_count = int(len(anomalous_unified_format_data) * validate_size)
    n_validate_count = int(len(normal_unified_format_data) * validate_size)
    # 验证集(正常/异常)样本个数
    a_test_count = int(len(anomalous_unified_format_data) * test_size)
    n_test_count = int(len(normal_unified_format_data) * test_size)

    # 切分训练集
    anomalous_train = anomalous_unified_format_data[:a_train_count]
    normal_train = normal_unified_format_data[:n_train_count]

    # 切分验证集
    anomalous_validate = anomalous_unified_format_data[a_train_count:a_train_count + a_validate_count]
    normal_validate = normal_unified_format_data[n_train_count:n_train_count + n_validate_count]

    # 切分测试集
    anomalous_test = anomalous_unified_format_data[-a_test_count:]
    normal_test = normal_unified_format_data[-n_test_count:]

    print("训练集异常样本个数总共：", len(anomalous_train))
    print("训练集正常样本个数总共：", len(normal_train))
    print("验证集异常样本个数总共：", len(anomalous_validate))
    print("验证集正常样本个数总共：", len(normal_validate))
    print("测试集异常样本个数总共：", len(anomalous_test))
    print("测试集正常样本个数总共：", len(normal_test))

    # 合并训练集、验证集、测试集数据
    x_train = np.row_stack((anomalous_train, normal_train))
    x_validate = np.row_stack((anomalous_validate, normal_validate))
    x_test = np.row_stack((anomalous_test, normal_test))

    print("训练集样本总数：", len(x_train))
    print("验证集样本总数：", len(x_validate))
    print("测试集样本总数：", len(x_test))

    # 生成标签
    y_train = np.append(np.ones(len(anomalous_train), dtype=np.int32),
                        np.zeros(len(normal_train), dtype=np.int32))
    y_validate = np.append(np.ones(len(anomalous_validate), dtype=np.int32),
                       np.zeros(len(normal_validate), dtype=np.int32))
    y_test = np.append(np.ones(len(anomalous_test), dtype=np.int32),
                       np.zeros(len(normal_test), dtype=np.int32))

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_validate shape:", x_validate.shape)
    print("y_validate shape:", y_validate.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # 洗牌
    x_train, y_train = shuffle(x_train, y_train, random_state=100)
    x_validate, y_validate = shuffle(x_validate, y_validate, random_state=100)
    x_test, y_test = shuffle(x_test, y_test, random_state=100)

    # 写入文件
    print('x_train, y_train, x_validate, y_validate, x_test, y_test写入文件...')
    pickle.dump(x_train, open('dataset/x_train.pkl', 'wb'))
    pickle.dump(y_train, open('dataset/y_train.pkl', 'wb'))
    pickle.dump(x_validate, open('dataset/x_validate.pkl', 'wb'))
    pickle.dump(y_validate, open('dataset/y_validate.pkl', 'wb'))
    pickle.dump(x_test, open('dataset/x_test.pkl', 'wb'))
    pickle.dump(y_test, open('dataset/y_test.pkl', 'wb'))
    print('x_train, y_train, x_validate, y_validate, x_test, y_test写入完成...')

    return (x_train, y_train), (x_validate, y_validate), (x_test, y_test)

# from keywords import get_keywords_dict
# def calc_high_freq_words():
#     # 创建一个预处理器
#     pre = PreProcessor(get_keywords_dict())
#     # 生成并保存异常样本 label=0
#     anomalous_unified_format_list = []
#     # 获取http文本列表
#     http_text_list = load_http_text_list('dataset/anomalousTrafficTest.txt')
#     http_text_list += load_http_text_list('dataset/normalTrafficTest.txt')
#     http_text_list += load_http_text_list('dataset/normalTrafficTraining.txt')
#     print('开始统计')
#     words_dict = {}
#     for http_text in http_text_list:
#         http_text = pre.url_decode(http_text)
#         # 标点符号左右添加空格
#         http_text = pre.addSpace(http_text)
#         # 以空格为分隔符划分单词
#         http_word_list = http_text.split(' ')
#         for word in http_word_list:
#             word = word.lower()
#             if word not in words_dict:
#                 words_dict[word] = 1
#                 print(word)
#             else:
#                 words_dict[word] += 1
#     pickle.dump(words_dict, open('word_dict_freq.pkl', 'wb'))
#     # words_dict = pickle.load(open('word_dict_freq.pkl', 'rb'))
#
#     words_list = []
#     for key in words_dict:
#         if words_dict[key] > 10:
#             words_list.append((words_dict[key], key))
#     # 排序
#     for i in range(len(words_list)):
#         for j in range(i, len(words_list) - 1):
#             if words_list[j][0] < words_list[j+1][0]:
#                 temp = words_list[j]
#                 words_list[j] = words_list[j+1]
#                 words_list[j+1] = temp
#     print(words_list)
#
#     # 生成txt
#     keywords_dict = get_keywords_dict()
#     with open('freq_txt.txt', 'w', encoding='utf-8') as f:
#         for _, word in words_list:
#             if str(word).isdigit():
#                 continue
#             if str(word).lower() in keywords_dict:
#                 continue
#             if len(word) < 3:
#                 continue
#             f.write(word.lower() + '\n')
