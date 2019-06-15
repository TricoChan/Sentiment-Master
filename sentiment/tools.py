# -*- coding: utf-8 -*-

import re
import jieba
from sklearn.externals import joblib
import numpy as np

import math
import pandas as pd

# 加载常用停用词
stopwords2 = [line.rstrip() for line in open('./哈工大停用词表.txt', 'r', encoding='utf-8')]
stopwords3 = [line.rstrip() for line in open('./四川大学机器智能实验室停用词库.txt', 'r', encoding='utf-8')]
stopwords = stopwords2 + stopwords3


def proc_text(raw_line):
    """
        处理每行的文本数据
        返回分词结果
    """
    # 1. 使用正则表达式去除非中文字符
    #    在 [] 内使用 ^ 表示非，否则表示行首
    filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
    # 将所有非中文字符替换为""
    chinese_only = filter_pattern.sub('', raw_line)
    # simplified_sentence = Converter('zh-hans').convert(chinese_only)

    # 2. 结巴分词+词性标注
    # 返回分词结果列表，包含单词和词性
    # words_lst = pseg.cut(chinese_only)
    words_lst = jieba.cut(chinese_only)

    # 3. 去除停用词
    # 将所有非停用词的词语存到列表里
    meaninful_words = []
    for word in words_lst:
        # if (word not in stopwords) and (flag == 'v'):
        # 也可根据词性去除非动词等
        if word not in stopwords:
            meaninful_words.append(word)

    # 返回一个字符串
    return ' '.join(meaninful_words)


def extract_feat_from_string(text, text_collection, common_words_freqs):
    common_words = [word for word, _ in common_words_freqs]
    feat_vec = []
    for word in common_words:
        # 如果高频词在当前text文本里，则计算TF-IDF值
        if word in text:
            tf_idf_val = text_collection.tf_idf(word, text)
        else:
            tf_idf_val = 0
        feat_vec.append(tf_idf_val)
    vec = np.array(feat_vec)
    return vec


# 词袋模型快速计算方法
def cal_bow(string):
    # 初始化string的文档向量
    document_vec = np.zeros(2000)
    # 加载字典
    dict = joblib.load('word_idf.pkl')
    # 清洗后的字符串
    cleaned_text = proc_text(string)
    # 清洗后的词列表
    word_list = cleaned_text.split(' ')
    # 计算词数
    word_count = len(word_list)
    # 去重后列表
    set_list = set(word_list)
    # 遍历去重后列表，从去重前列表读取词频
    for item in set_list:
        try:
            tf_idf = word_list.count(item) / word_count * dict[item][1]
            document_vec[dict[item][0]] = tf_idf
        except:
            continue
    return document_vec


# 字典获取方法
def get_dict():
    dict = {}
    text_collection = joblib.load('NB_text_collection_2000.pkl')
    common_words_freqs = joblib.load('NB_common_words_2000.pkl')
    common_words = [word for word, _ in common_words_freqs]
    i = 0
    # 字典中key对应的list，0存索引，1存idf
    for word in common_words:
        list = []
        list.append(i)
        list.append(text_collection.idf(word))
        a = {word: list}
        dict.update(a)
        i = i + 1
    print(dict)
    joblib.dump(dict, 'word_idf.pkl')


def string2bow(string):
    # 重新加载数据
    text_collection = joblib.load('NB_text_collection_2000.pkl')
    common_words_freqs = joblib.load('NB_common_words_2000.pkl')
    # 构建词向量
    pro = proc_text(string)
    text_vec = pro.split(' ')
    cbow = extract_feat_from_string(text_vec, text_collection, common_words_freqs)
    return cbow


def split_train_test(text_df, size=0.8):
    """
        分割训练集和测试集
        size = 0.8
        表示按二八法则分隔数据集，80%做为训练集，20%测试集
    """
    # 为保证每个类中的数据能在训练集中和测试集中的比例相同，所以需要依次对每个类进行处理
    train_text_df = pd.DataFrame()
    test_text_df = pd.DataFrame()

    # 表示情感值
    labels = [0, 1]
    for label in labels:
        # 找出label的记录
        text_df_w_label = text_df[text_df['label'] == label]
        # 重新设置索引，保证每个类的记录是从0开始索引，方便之后的拆分
        text_df_w_label = text_df_w_label.reset_index()

        # 默认按80%训练集，20%测试集分割
        # 这里为了简化操作，取前80%放到训练集中，后20%放到测试集中
        # 当然也可以随机拆分80%，20%（尝试实现下DataFrame中的随机拆分）

        # 该类数据的行数
        n_lines = text_df_w_label.shape[0]
        # 16432 * 0.8 = 8000
        # 根据size值，获取训练集的行数 math.floor() 求浮点数向下最接近的整数
        split_line_no = math.floor(n_lines * size)

        # 取出当前类的文本的 开始 ~split_line_no 部分行做为训练集
        text_df_w_label_train = text_df_w_label.iloc[:split_line_no, :]
        # 取出当前类的文本的 split_line_no ~ 最后 部分行做为测试集
        text_df_w_label_test = text_df_w_label.iloc[split_line_no:, :]

        # 放入整体训练集，测试集中
        train_text_df = train_text_df.append(text_df_w_label_train)
        test_text_df = test_text_df.append(text_df_w_label_test)

    # 重置索引
    train_text_df = train_text_df.reset_index()
    # 重置索引
    test_text_df = test_text_df.reset_index()
    # 包含所有的训练集 和 测试集
    return train_text_df, test_text_df


def text2vec(text_list):
    # np.set_printoptions(threshold = 1e6)
    vectors = np.load('top10000_vectors.npy')
    # 训练集
    train_set_list = []
    for line in text_list:
        # 仅保留前60个词
        i = 0
        word_vec = []
        # 长度为60词，词向量维度300的句向量
        sentence_vec = np.zeros((60, 300))
        word_list = line.split()
        # print(word_list)
        for word in word_list:
            if i > 59:
                break
            try:
                # 若存在
                word_vec = vectors[word]
            except:
                # 随机生成[-1,1]的随机数
                word_vec = np.random.rand(300) * 2 - 1
            sentence_vec[i] = word_vec
            i += 1
        train_set_list.append(sentence_vec)
    train_set = np.array(train_set_list)
    return train_set
