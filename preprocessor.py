# Name: preprocessor.py
# Author: HuangHao
# Time: 2021/2/28 6:19

import urllib.parse


class PreProcessor:
    """
    文本数据预处理器，将文本数据格式统一化
    """

    def __init__(self, keywords_dict):
        """
        构造函数
        :param keywords_dict:
        """
        # 符号分隔符，其中特殊的是换行符\n
        self.symbol = ['`', '\\', '-', '=', '[',  ']', ';', '\'', ',', '.', '/', '~', '@', '#', '$', '%',
                       '^', '&', '*', '(', ')', '_', '+', '{', '}', '|', ':', '"', '<', '>', '?', '\n']
        self.keywords_dict = keywords_dict
        self.index_keywords_dict = dict(zip(self.keywords_dict.values(), self.keywords_dict.keys()))

    def url_decode(self, http_text):
        """
        url解码，将url编码解码成ASCCI码
        :param http_text: http文本
        :return: http_text
        """
        return urllib.parse.unquote(http_text, encoding='utf-8')

    def addSpace(self, http_text, symbol=None):
        return self.__add_space(http_text, symbol)

    def __add_space(self, http_text, symbol=None):
        """
        在标点符号两边添加空格,包括换行符号也是标点符号
        :param http_text:
        :return:
        """
        if symbol is None:
            symbol = self.symbol
        ret_text = ''
        # 上一个字符是否为标点符号
        pre_s_is_symbol = False
        for s in http_text:
            # 如果这个字符是标点符号,准备加空格
            if s in symbol:
                # 如果上个字符不是标点符号，那么两边加空格
                if not pre_s_is_symbol:
                    ret_text += ' ' + s + ' '
                # 如果上个字符是标点符号，那么只是右边加空格
                else:
                    ret_text += s + ' '
                pre_s_is_symbol = True
            else:
                ret_text += s
                pre_s_is_symbol = False
        return ret_text

    def __replace_keywords(self, http_word_list):
        """
        替换关键词，传入单词列表，替换
        :param http_word_list:
        :return: http_word_list
        """
        http_keywords_list = []
        for word in http_word_list:
            word = word.lower()
            if word in self.keywords_dict:
                http_keywords_list.append(word)
            # 如果单词是回车符号,替换为newline
            elif word == '\n':
                http_keywords_list.append('newline')
            # 如果单词是全字母，替换为purestring
            elif word.isalpha():
                http_keywords_list.append('purestring')
            # 如果单词是全数字，替换为numbers
            elif word.isdigit():
                http_keywords_list.append('numbers')
            # 如果单词是混合字符串，替换为numbers
            else:
                http_keywords_list.append('mixstring')
        return http_keywords_list

    def format_unification_wordlist(self, http_text):
        """
        格式统一化，返回统一化后的单词列表
        :param http_text:
        :return:
        """
        # url解码
        http_text = self.url_decode(http_text)
        # 标点符号左右添加空格
        http_text = self.__add_space(http_text)
        # 以空格为分隔符划分单词
        http_word_list = http_text.split(' ')
        # 替换关键词
        http_word_list = self.__replace_keywords(http_word_list)
        return http_word_list

    def format_unification_text(self, http_text):
        """
        格式统一化，输出格式统一化后的文本
        :param http_text:
        :return:
        """
        http_word_list = self.format_unification_wordlist(http_text)
        return ' '.join(http_word_list)

    def format_unification_digital_list(self, http_text):
        """
        格式统一化，输出格式统一化后的数字数组
        :param http_text:
        :return:
        """
        http_digital_list = []
        http_word_list = self.format_unification_wordlist(http_text)
        for word in http_word_list:
            if word in self.keywords_dict:
                http_digital_list.append(self.keywords_dict[word])
        return http_digital_list

    def digital_to_words(self, digital_list):
        """
        数字数组转单词数组
        :param digital_list:
        :return:
        """
        words_list = []
        for digital in digital_list:
            # 如果digital是填充符，就结束转换
            if digital == 0:
                break
            words_list.append(self.index_keywords_dict[digital])
        return words_list


