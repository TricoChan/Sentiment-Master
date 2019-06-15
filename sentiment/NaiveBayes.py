import os
import pandas as pd
import nltk
import datetime
from tools import proc_text, split_train_test
from nltk.text import TextCollection
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import predict

dataset_path = './dataset'

# 原始数据的csv文件
output_text_filename = 'raw_weibo_text.csv'

# 清洗好的文本数据文件
output_cln_text_filename = 'clean_weibo_text.csv'

# 删除行后的文件
delete_filename = 'delete_text.csv'


# 时间差计算函数
def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1


def run_main():
    # 删除行
    '''
    text_df = pd.read_csv(os.path.join(dataset_path, output_text_filename),
                      encoding='utf-8')
    df = text_df.drop(labels=range(106728,159814),axis=0)
    df.to_csv(os.path.join(dataset_path, output_cln_text_filename),
                   index=None, encoding='utf-8')
    '''
    # 修改分类
    '''
    text_df = pd.read_csv(os.path.join(dataset_path, output_cln_text_filename),
                         encoding='utf-8')
    text_df['label'].replace(4,0,inplace=True)
    text_df.to_csv(os.path.join(dataset_path, output_cln_text_filename),
                   index=None, encoding='utf-8')


    # 输出去停用词前数据
    text_df = pd.read_csv(os.path.join(dataset_path, output_cln_text_filename),
                          encoding='utf-8')
    print(text_df.head(5))
    '''

    '''
    # 去停用词
    text_df = pd.read_csv(os.path.join(dataset_path, output_cln_text_filename),
                          encoding='utf-8')
    text_df['text'] = text_df['text'].apply(proc_text)

    # 过滤空字符串，去掉所有空行部分
    text_df = text_df[text_df['text'] != '']

    # 保存处理好的文本数据，文本预处理结束
    text_df.to_csv(os.path.join(dataset_path, output_cln_text_filename),
                   index=None, encoding='utf-8')
    print(text_df.head(5))
    print('完成，并保存结果。')
    '''

    # 训练集划分------------------------------------------------------------
    print('加载处理好的文本数据')
    clean_text_df = pd.read_csv(os.path.join(dataset_path, output_cln_text_filename),
                                encoding='utf-8')
    # 分割训练集和测试集
    # 按每个情感值的80%做分割，
    train_text_df, test_text_df = split_train_test(clean_text_df)
    # 查看训练集测试集基本信息
    print('训练集中各类的数据个数：', train_text_df.groupby('label').size())
    print('测试集中各类的数据个数：', test_text_df.groupby('label').size())
    # -------------------------------------------------------------------------------

    # 构建词袋模型----------------------------------------------------------------------------
    clean_text_df = pd.read_csv(os.path.join(dataset_path, output_cln_text_filename),
                                encoding='utf-8')
    train_text_df, test_text_df = split_train_test(clean_text_df)
    # 计算词频
    n_common_words = 2000
    # 将训练集中的单词拿出来统计词频
    print('统计词频...')
    # 获取训练集数据集里所有的词语的列表
    all_words_in_train = get_word_list_from_data(train_text_df)
    print(all_words_in_train)
    # 统计词频
    fdisk = nltk.FreqDist(all_words_in_train)
    # fdisk.plot(5000)
    # 获取词频排名前300个的词语的词频
    # 构建“常用单词列表”
    common_words_freqs = fdisk.most_common(n_common_words)
    print('出现最多的{}个词是：'.format(n_common_words))
    for word, count in common_words_freqs:
        print('{}: {}次'.format(word, count))
    print()
    # 在训练集上提取特征
    # 将text部分转换为list做为参数
    text_collection = TextCollection(train_text_df['text'].values.tolist())
    # 提取训练样本和测试样本的特征
    # _X 表示常用单词在每一行的tf-idf值，_y 表示情感值
    print('训练样本提取特征...', end=' ')
    train_X, train_y = extract_feat_from_data(train_text_df, text_collection, common_words_freqs)
    print('完成')
    print()

    print('测试样本提取特征...', end=' ')
    test_X, test_y = extract_feat_from_data(test_text_df, text_collection, common_words_freqs)
    print('完成')
    # -------------------------------------------------------------------------------

    # 高斯贝叶斯模型---------------------------------------------------------------------------------
    print('训练模型...', end=' ')
    # 创建高斯朴素贝叶斯模型
    gnb = GaussianNB()

    # 向模型加载训练集特征数据，训练模型，
    gnb.fit(train_X, train_y)
    # 保存模型相关数据
    joblib.dump(gnb, 'NaiveBayes_2000.pkl')
    joblib.dump(text_collection, 'NB_text_collection_2000.pkl')
    joblib.dump(common_words_freqs, 'NB_common_words_2000.pkl')

    print('完成')
    print()
    # ---------------------------------------------------------------------------------

    # 模型评估---------------------------------------------------------------------------------
    test_pred = gnb.predict(test_X)
    print('准确率：', cal_acc(test_y, test_pred))
    # ---------------------------------------------------------------------------------
    predict.NB_predict('电池挺好的，但是续航不行')


if __name__ == '__main__':
    run_main()
