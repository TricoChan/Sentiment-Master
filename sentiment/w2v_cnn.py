import numpy as np
import numpy as np
import pandas as pd
import jieba
import os
import datetime
import re
import math
from NaiveBayes import subtime
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Model, load_model
from NaiveBayes import output_cln_text_filename
from tools import split_train_test
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras import regularizers
from sklearn.externals import joblib

K.clear_session()
dataset_path = './train_set'
testset_path = './test_set'
clean_path = './dataset'


def load_train_vectors():
    while True:
        # 构建每个batch
        for i in range(1, 3320):
            train_set_path = os.path.join(dataset_path, 'train_set_' + str(i) + '.npy')
            train_label_path = os.path.join(dataset_path, 'train_lable_' + str(i) + '.npy')
            train_set_slice = np.load(train_set_path)
            train_lable_slice = np.load(train_label_path)
            # train_set_array = np.concatenate((train_set_array, train_set_slice), axis=0)
            # train_lable_array = np.concatenate((train_label_array, train_lable_slice), axis=0)
            yield train_set_slice, train_lable_slice
            # print('加载了第' + str(i) + '轮的数据')


def load_test_vectors():
    while True:
        # 构建每个batch
        for i in range(1, 51):
            test_set_path = os.path.join(testset_path, 'test_set_' + str(i) + '.npy')
            test_label_path = os.path.join(testset_path, 'test_lable_' + str(i) + '.npy')
            test_set_slice = np.load(test_set_path)
            test_lable_slice = np.load(test_label_path)
            yield test_set_slice, test_lable_slice


def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim


def text2vec(text_list):
    # np.set_printoptions(threshold = 1e6)
    vectors_path = "sgns.weibo.char"
    vectors, iw, wi, dim = read_vectors(vectors_path, 10000)
    # matrix, iw, wi = load_matrix(vectors_path)
    # print(matrix[wi['好']])
    #print('词向量加载成功')
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
                print('长度超限值')
                break
            try:
                # 若存在
                word_vec = vectors[word]
            except:
                # 随机生成[-1,1]的随机数
                word_vec = np.random.rand(300) * 2 - 1
            sentence_vec[i] = word_vec
            i += 1
        # print(sentence_vec)
        # print(sentence_vec.shape)
        train_set_list.append(sentence_vec)
    train_set = np.array(train_set_list)
    return train_set
    # print(train_set.shape)
    # print(train_set[0])


def text_cnn(maxlen=60, max_features=10000, embed_size=300):
    pretrained_sentence_vectors = Input(shape=(60, 300))
    convs = []
    sizes = [2, 3, 4, 5]
    for sz in sizes:
        l_conv = Conv1D(filters=100, kernel_size=sz, activation='relu')(pretrained_sentence_vectors)
        l_pool = MaxPooling1D(maxlen - sz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    out = Dropout(0.5)(merge)
    allcon = Dense(64, activation='relu')(out)
    # l_batch = BatchNormalization()(output)
    output = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.1))(allcon)
    model = Model(pretrained_sentence_vectors, output)
    adam = optimizers.Adam(lr=0.00001)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])
    return model


def run_main():
    # np.set_printoptions(threshold=1e6)

    # ----------------------------------------------------------  1.加载清洗后的数据集和测试集
    # # 加载清洗后的数据文件
    # clean_text_df = pd.read_csv(os.path.join(clean_path, output_cln_text_filename),
    #                             encoding='utf-8')
    # # 划分训练集和测试集
    # train, test = split_train_test(clean_text_df, 0.8)
    # train_text_list = []
    # train_label_list = []
    # test_text_list = []
    # test_label_list = []
    # print('构建数据集中')
    # for i, r_data in train.iterrows():
    #     train_text_list.append(r_data['text'])
    #     train_label_list.append(r_data['label'])
    # for i, r_data in test.iterrows():
    #     test_text_list.append(r_data['text'])
    #     test_label_list.append(r_data['label'])
    # # 打乱顺序
    # c = list(zip(train_text_list, train_label_list))
    # np.random.shuffle(c)
    # train_text_list[:], train_label_list[:] = zip(*c)
    # c = list(zip(test_text_list, test_label_list))
    # np.random.shuffle(c)
    # test_text_list[:], test_label_list[:] = zip(*c)
    # joblib.dump(train_text_list, 'train_set.pkl')
    # joblib.dump(test_text_list, 'test_set.pkl')
    # joblib.dump(train_label_list, 'train_label.pkl')
    # joblib.dump(test_label_list, 'test_label.pkl')

    # ----------------------------------------------------------  2.分批使用word2vec构建文本表示
    # train_text_list = joblib.load('train_set.pkl')
    # test_text_list = joblib.load('test_set.pkl')
    # train_label_list = joblib.load('train_label.pkl')
    # test_label_list = joblib.load('test_label.pkl')
    # # ----------------------------------------------------------  训练集
    # print('构建词向量中')
    # total = len(train_text_list)
    # # 212417条数据 3319个batch 每个batch64个数据
    # num = math.floor(total / 3319)
    # index = 0
    # for i in range(1, 3320):
    #     tail = index + num
    #     train_set = text2vec(train_text_list[index:tail])
    #     train_lable = train_label_list[index:tail]
    #     np.save(os.path.join(dataset_path, 'train_set_' + str(i)), train_set)
    #     np.save(os.path.join(dataset_path, 'train_lable_' + str(i)), train_lable)
    #     index = tail
    # ----------------------------------------------------------  测试集
    # total = len(test_text_list)
    # # 53104条数据 50个batch 每个batch1062个数据
    # num = math.floor(total / 50)
    # index = 0
    # for i in range(1, 51):
    #     tail = index + num
    #     test_set = text2vec(test_text_list[index:tail])
    #     test_lable = test_label_list[index:tail]
    #     np.save(os.path.join(testset_path, 'test_set_' + str(i)), test_set)
    #     np.save(os.path.join(testset_path, 'test_lable_' + str(i)), test_lable)
    #     index = tail
    # print('构建完成')

    # ----------------------------------------------------------  3.批量训练
    print('构建TEXT-CNN模型中')
    model = text_cnn()
    # 获取当前时间
    startdate = datetime.datetime.now()
    # 当前时间转换为指定字符串格
    startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")
    history = model.fit_generator(load_train_vectors(),
                                  epochs=50,
                                  steps_per_epoch=3319,
                                  validation_steps=50,
                                  validation_data=load_test_vectors())
    enddate = datetime.datetime.now()
    enddate = enddate.strftime("%Y-%m-%d %H:%M:%S")
    print('训练时长', subtime(startdate, enddate))
    model.save('text_cnn_w2v.h5')
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # model = load_model('text_cnn_finally.h5')
    # scores = model.evaluate(x_test, y_test)
    # print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

    # a = np.load(os.path.join(dataset_path, 'train_set_3200.npy'))
    # print(a.shape)


if __name__ == '__main__':
    run_main()
