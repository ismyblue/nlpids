# Name: __init__.py
# Author: HuangHao
# Time: 2021/3/8 15:00

"""
语料库：包含各协议、编程语言、标点符号、高频词汇等单词。可通过此py文件获取一个字典对象，key为语料，value为词索引
"""
import pickle
import os


def keywords_dict_size():
    if os.path.exists('./keywords/keywords_dict_size.pickle'):
        return pickle.load(open('./keywords/keywords_dict_size.pickle', 'rb'))
    else:
        get_keywords_dict()
        return pickle.load(open('./keywords/keywords_dict_size.pickle', 'rb'))


def get_keywords_dict():
    """
    获取关键词字典，keywords_dict,key为语料，value为词索引
    :return: keywords_dict
    """
    # 如果已经存在字典对象，那么直接返回这个字典对象
    # if os.path.exists('./keywords/keywords_dict.pickle'):
    #     return pickle.load(open('./keywords/keywords_dict.pickle', 'rb'))
    # 如果没有keywords_dict.pickle文件，即没有字典对象，那么重新构造一个
    # 读取关键词txt文件，加入集合去重，加入字典
    keywords_dict = {'_PLACEHOLDER_': 0, 'newline': 1, 'numbers': 2, 'purestring': 3, 'mixstring': 4}
    index = 5
    files = os.listdir('./keywords')
    for file in files:
        txt = os.path.join('./keywords', file)
        # 处理一个txt文件
        if txt.endswith('.txt') and os.path.isfile(txt):
            # 读取txt文件中的关键词
            f = open(txt, 'r', encoding='utf-8')
            # 读取一行关键词
            lines = f.readlines()
            for line in lines:
                # 切分每行关键词
                keywords = line.split()
                # 每个关键词加入字典
                for keyword in keywords:
                    keyword = keyword.lower()
                    if keyword in keywords_dict:
                        continue
                    keywords_dict[keyword] = index
                    index += 1
    # 关键词字典持久化，保存为文件 keywords_dict.pickle
    pickle.dump(keywords_dict, open('./keywords/keywords_dict.pickle', 'wb'))
    print('字典大小{}'.format(len(keywords_dict)))
    pickle.dump(len(keywords_dict), open('./keywords/keywords_dict_size.pickle', 'wb'))
    return keywords_dict










