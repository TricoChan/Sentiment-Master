import numpy as np
import pandas as pd
import jieba
import os
import datetime
from NaiveBayes import subtime
import re
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Model,load_model
from NaiveBayes import dataset_path, output_cln_text_filename
from tools import split_train_test
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras import regularizers
from sklearn.externals import joblib

K.clear_session()


def text_cnn(maxlen=100, max_features=10000, embed_size=300):
    comment_seq = Input(shape=[maxlen])
    emb_comment = Embedding(max_features, embed_size)(comment_seq)
    convs = []
    sizes = [2, 3, 4, 5]
    for sz in sizes:
        l_conv = Conv1D(filters=100, kernel_size=sz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - sz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    out = Dropout(0.5)(merge)
    allcon = Dense(32, activation='relu')(out)
    # l_batch = BatchNormalization()(output)
    output = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.1))(allcon)
    model = Model([comment_seq], output)
    adam = optimizers.Adam(lr=0.00001)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])
    return model


def run_main():
    # 加载清洗后的数据文件
    clean_text_df = pd.read_csv(os.path.join(dataset_path, output_cln_text_filename),
                                encoding='utf-8')
    # 划分训练集和测试集
    train, test = split_train_test(clean_text_df, 0.8)
    train_text_list = []
    train_label_list = []
    test_text_list = []
    test_label_list = []
    print('构建数据集中')
    for i, r_data in train.iterrows():
        train_text_list.append(r_data['text'])
        train_label_list.append(r_data['label'])
    for i, r_data in test.iterrows():
        test_text_list.append(r_data['text'])
        test_label_list.append(r_data['label'])

    print(train_text_list)
    print(type(train_text_list))
    print(train_text_list[0])

    # 打乱顺序
    c = list(zip(train_text_list,train_label_list ))
    np.random.shuffle(c)
    train_text_list[:], train_label_list[:] = zip(*c)
    c = list(zip(test_text_list,test_label_list ))
    np.random.shuffle(c)
    test_text_list[:], test_label_list[:] = zip(*c)


    print('构建词向量中')
    # 建立2000个单词的字典
    tokenizer = Tokenizer(num_words=10000)
    # 读取所有训练集文本，词频排序TOP2000会被列入字典
    tokenizer.fit_on_texts(train_text_list)
    joblib.dump(tokenizer,'tokenizer.pkl')
    #
    # 将训练集和测试集文本转为数字序列
    x_train_seq = tokenizer.texts_to_sequences(train_text_list)
    x_test_seq = tokenizer.texts_to_sequences(test_text_list)
    # 截长补短
    x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=100)
    y_train = np.array(train_label_list).reshape(-1,1)
    y_test = np.array(test_label_list).reshape(-1,1)
    # print('数字序列：{0}'.format(x_train_seq))
    # print('数字序列类型：{0}'.format(type(x_train_seq)))
    # print('截断后：{0}'.format(x_train))
    # print('类型：{0}'.format(type(x_train)))
    # print(x_train_seq[0])
    # print(x_train[0])


    print('构建TEXT-CNN模型中')
    model = text_cnn()
    batch_size = 64
    epochs = 60
    # 获取当前时间
    startdate = datetime.datetime.now()
    # 当前时间转换为指定字符串格
    startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")
    history = model.fit(x_train, y_train,
              validation_split=0.25,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)
    enddate = datetime.datetime.now()
    enddate = enddate.strftime("%Y-%m-%d %H:%M:%S")
    print('训练时长', subtime(startdate, enddate))
    model.save('text_cnn_alter.h5')
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

    model = load_model('text_cnn_finally.h5')
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

if __name__ == '__main__':
    run_main()
