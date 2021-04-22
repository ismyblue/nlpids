# Name: __init__.py
# Author: HuangHao
# Time: 2021/3/8 19:33

import tensorflow as tf
from preprocessor import PreProcessor
from keywords import get_csic2010_keywords_dict, csic2010_keywords_dict_size, get_homemade_dataset_keywords_dict, \
    homemade_dataset_keywords_dict_size
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


def split_dataset(unified_format_data, sample_count_list, validate_size=0.2, test_size=0.2):
    """
    划分数据集
    :param unified_format_data: 填充了长度的格式统一的样本
    :param sample_count_list: 各种样本的个数的列表
    :param validate_size: 验证集比例
    :param test_size: 测试集比例
    :return:
    """
    # 洗牌
    # 各种类数据的列表
    data_list = [[] for _ in range(len(sample_count_list))]
    # 训练集合中各种类的个数
    train_count_list = []
    # 验证集合中各种类的个数
    validate_count_list = []
    # 测试集合中各种类的个数
    test_count_list = []
    pre = 0
    for label, count in enumerate(sample_count_list):
        data_list[label] = shuffle(unified_format_data[pre:pre+count], random_state=100)
        pre += count
        # 训练集中数据个数
        train_count_list.append(int(count * (1 - test_size - validate_size)))
        # 验证集中数据个数
        validate_count_list.append(int(count * validate_size))
        # 测试集中数据个数
        test_count_list.append(int(count * test_size))

    # 生成训练集、验证集、测试集数据
    x_train = data_list[0][:train_count_list[0]]
    x_validate = data_list[0][train_count_list[0]:train_count_list[0]+validate_count_list[0]]
    x_test = data_list[0][-test_count_list[0]:]
    for label in range(1, len(sample_count_list)):
        x_train = np.row_stack((x_train, data_list[label][:train_count_list[label]]))
        x_validate = np.row_stack((x_validate, data_list[label][train_count_list[label]:train_count_list[label]+validate_count_list[label]]))
        x_test = np.row_stack((x_test, data_list[label][-test_count_list[label]:]))
    # 训练集,验证集，测试集
    print("训练集样本个数分别为：", train_count_list)
    print("验证集样本个数分别为：", validate_count_list)
    print("测试集样本个数分别为：", validate_count_list)
    print("训练集样本总数：", x_train.shape[0])
    print("验证集样本总数：", x_validate.shape[0])
    print("测试集样本总数：", x_test.shape[0])

    # 生成标签
    y_train = np.zeros(train_count_list[0], dtype=np.int32)
    y_validate = np.zeros(validate_count_list[0], dtype=np.int32)
    y_test = np.zeros(validate_count_list[0], dtype=np.int32)
    for label in range(1, len(train_count_list)):
        y_train = np.append(y_train, np.array([label] * train_count_list[label], dtype=np.int32))
        y_validate = np.append(y_validate, np.array([label] * validate_count_list[label], dtype=np.int32))
        y_test = np.append(y_test, np.array([label] * validate_count_list[label], dtype=np.int32))

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

    return (x_train, y_train), (x_validate, y_validate), (x_test, y_test)


def load_csic_2010():
    """
    载入http dataset csis 2010数据集，
    :return: (x_train, y_train), (x_validate, y_validate), (x_test, y_test)
    """
    # 获取预处理后的样本list
    # 如果已经有序列化好的list对象，那么直接读取
    if os.path.exists('dataset/csic2010/x_train.pkl') \
            and os.path.exists('dataset/csic2010/y_train.pkl') \
            and os.path.exists('dataset/csic2010/x_validate.pkl') \
            and os.path.exists('dataset/csic2010/y_validate.pkl') \
            and os.path.exists('dataset/csic2010/x_test.pkl') \
            and os.path.exists('dataset/csic2010/y_test.pkl'):
        # 读取序列化对象
        x_train = pickle.load(open('dataset/csic2010/x_train.pkl', 'rb'))
        y_train = pickle.load(open('dataset/csic2010/y_train.pkl', 'rb'))
        x_validate = pickle.load(open('dataset/csic2010/x_validate.pkl', 'rb'))
        y_validate = pickle.load(open('dataset/csic2010/y_validate.pkl', 'rb'))
        x_test = pickle.load(open('dataset/csic2010/x_test.pkl', 'rb'))
        y_test = pickle.load(open('dataset/csic2010/y_test.pkl', 'rb'))
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_validate shape:", x_validate.shape)
        print("y_validate shape:", y_validate.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)
        return (x_train, y_train), (x_validate, y_validate), (x_test, y_test)
    # 如果没有序列化好的list对象，那么就预处理然后生成
    else:
        if os.path.exists('dataset/csic2010/normalUnifiedFormat.pkl') and os.path.exists(
                'dataset/csic2010/normalUnifiedFormat.pkl'):
            normal_unified_format_list = pickle.load(open('dataset/csic2010/normalUnifiedFormat.pkl', 'rb'))
            anomalous_unified_format_list = pickle.load(open('dataset/csic2010/anomalousUnifiedFormat.pkl', 'rb'))
        else:
            # 创建一个预处理器
            pre = PreProcessor(get_csic2010_keywords_dict())

            # 生成并保存正常样本 label=1
            normal_unified_format_list = []
            # 获取http文本列表
            http_text_list = load_http_text_list('dataset/csic2010/normalTrafficTest.txt')
            http_text_list += load_http_text_list('dataset/csic2010/normalTrafficTraining.txt')
            # 进行预处理，生成统一格式数据
            for http_text in http_text_list:
                normal_unified_format_list.append(pre.format_unification_digital_list(http_text))
            pickle.dump(normal_unified_format_list, open('dataset/csic2010/normalUnifiedFormat.pkl', 'wb'))

            # 生成并保存异常样本 label=0
            anomalous_unified_format_list = []
            # 获取http文本列表
            http_text_list = load_http_text_list('dataset/csic2010/anomalousTrafficTest.txt')
            # 进行预处理，生成统一格式数据
            for http_text in http_text_list:
                anomalous_unified_format_list.append(pre.format_unification_digital_list(http_text))
            pickle.dump(anomalous_unified_format_list, open('dataset/csic2010/anomalousUnifiedFormat.pkl', 'wb'))

    # 样本数据
    data = np.append(np.array(normal_unified_format_list, dtype=object),
                     np.array(anomalous_unified_format_list, dtype=object))

    # 可变长度填充为定长
    data = tf.keras.preprocessing.sequence.pad_sequences(data, padding="post")
    # 切分数据集,洗牌，并保存
    (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = split_dataset(
        data, [len(normal_unified_format_list), len(anomalous_unified_format_list)], validate_size=0.2,
        test_size=0.2)
    # 写入文件
    print('x_train, y_train, x_validate, y_validate, x_test, y_test写入文件...')
    pickle.dump(x_train, open('dataset/csic2010/x_train.pkl', 'wb'))
    pickle.dump(y_train, open('dataset/csic2010/y_train.pkl', 'wb'))
    pickle.dump(x_validate, open('dataset/csic2010/x_validate.pkl', 'wb'))
    pickle.dump(y_validate, open('dataset/csic2010/y_validate.pkl', 'wb'))
    pickle.dump(x_test, open('dataset/csic2010/x_test.pkl', 'wb'))
    pickle.dump(y_test, open('dataset/csic2010/y_test.pkl', 'wb'))
    print('x_train, y_train, x_validate, y_validate, x_test, y_test写入完成...')
    return (x_train, y_train), (x_validate, y_validate), (x_test, y_test)


def load_homemade_dataset():
    """
    载入homemade dataset数据集，
    :return: (x_train, y_train), (x_validate, y_validate), (x_test, y_test)
    """
    # 获取预处理后的样本list
    # 如果已经有序列化好的list对象，那么直接读取
    if os.path.exists('dataset/homemade_dataset/x_train.pkl') \
            and os.path.exists('dataset/homemade_dataset/y_train.pkl') \
            and os.path.exists('dataset/homemade_dataset/x_validate.pkl') \
            and os.path.exists('dataset/homemade_dataset/y_validate.pkl') \
            and os.path.exists('dataset/homemade_dataset/x_test.pkl') \
            and os.path.exists('dataset/homemade_dataset/y_test.pkl'):
        # 读取序列化对象
        x_train = pickle.load(open('dataset/homemade_dataset/x_train.pkl', 'rb'))
        y_train = pickle.load(open('dataset/homemade_dataset/y_train.pkl', 'rb'))
        x_validate = pickle.load(open('dataset/homemade_dataset/x_validate.pkl', 'rb'))
        y_validate = pickle.load(open('dataset/homemade_dataset/y_validate.pkl', 'rb'))
        x_test = pickle.load(open('dataset/homemade_dataset/x_test.pkl', 'rb'))
        y_test = pickle.load(open('dataset/homemade_dataset/y_test.pkl', 'rb'))
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
        pre = PreProcessor(get_homemade_dataset_keywords_dict())
        # 样本文件名
        sample_file_names = ['normal.txt', 'sqli.txt', 'xss.txt', 'fi.txt', 'rce.txt']
        # label分别为: 0, 1, 2, 3, 4

        # 生成并保存样本
        # 所有样本列表
        data = np.array([])
        # 各种样本的个数
        sample_count_list = []
        for label, sample_file in enumerate(sample_file_names):
            # 生成并保存样本
            unified_format_list = []
            # 获取http文本列表
            http_text_list = load_http_text_list('dataset/homemade_dataset/' + sample_file)
            # 进行预处理，生成统一格式数据
            for http_text in http_text_list:
                unified_format_list.append(pre.format_unification_digital_list(http_text))
            # 保存样本个数
            sample_count_list.append(len(unified_format_list))
            # 保存样本文件
            pickle.dump(unified_format_list, open('dataset/homemade_dataset/' + sample_file.split('.')[0] + '.pkl', 'wb'))
            if label == 0:
                data = np.array(np.array(unified_format_list, dtype=object))
            else:
                data = np.append(data, np.array(unified_format_list, dtype=object))
    # 可变长度填充为定长
    data = tf.keras.preprocessing.sequence.pad_sequences(data, padding="post")

    # 切分数据集,洗牌，并保存
    (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = split_dataset(
        data, sample_count_list, validate_size=0.2, test_size=0.2)
    # 写入文件
    print('x_train, y_train, x_validate, y_validate, x_test, y_test写入文件...')
    pickle.dump(x_train, open('dataset/homemade_dataset/x_train.pkl', 'wb'))
    pickle.dump(y_train, open('dataset/homemade_dataset/y_train.pkl', 'wb'))
    pickle.dump(x_validate, open('dataset/homemade_dataset/x_validate.pkl', 'wb'))
    pickle.dump(y_validate, open('dataset/homemade_dataset/y_validate.pkl', 'wb'))
    pickle.dump(x_test, open('dataset/homemade_dataset/x_test.pkl', 'wb'))
    pickle.dump(y_test, open('dataset/homemade_dataset/y_test.pkl', 'wb'))
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
