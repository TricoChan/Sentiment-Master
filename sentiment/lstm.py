import numpy as np
import pandas as pd
import jieba
import os
import datetime
import matplotlib.pyplot as plt
from keras.utils import plot_model
import re
from NaiveBayes import subtime
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from NaiveBayes import dataset_path, output_cln_text_filename
from tools import split_train_test
from keras import backend as K
from keras import regularizers,optimizers
K.clear_session()



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

    # 打乱顺序
    c = list(zip(train_text_list, train_label_list))
    np.random.shuffle(c)
    train_text_list[:], train_label_list[:] = zip(*c)
    c = list(zip(test_text_list, test_label_list))
    np.random.shuffle(c)
    test_text_list[:], test_label_list[:] = zip(*c)


    print('构建词向量中')
    # 建立2000个单词的字典
    tokenizer = Tokenizer(num_words=4000)
    # 读取所有训练集文本，词频排序TOP2000会被列入字典
    tokenizer.fit_on_texts(train_text_list)
    # 将训练集和测试集文本转为数字序列
    x_train_seq = tokenizer.texts_to_sequences(train_text_list)
    x_test_seq = tokenizer.texts_to_sequences(test_text_list)
    # 截长补短
    x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=100)

    y_train = train_label_list
    y_test = test_label_list

    print('构建LSTM模型中')
    model = Sequential()
    model.add(Embedding(4000, 200))
    model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(1, activation='sigmoid'))
    batch_size = 64
    epochs = 50
    adam = optimizers.adam(lr= 0.0001)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    # 获取当前时间
    startdate = datetime.datetime.now()
    # 当前时间转换为指定字符串格
    startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")

    history = model.fit(x_train, y_train,
              validation_split=0.1,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)

    enddate = datetime.datetime.now()
    enddate = enddate.strftime("%Y-%m-%d %H:%M:%S")
    print('LSTM训练时长', subtime(startdate, enddate))

    model.save('lstm_finally.h5')

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

    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    plot_model(model, to_file='model.png')

if __name__ == '__main__':
    run_main()
