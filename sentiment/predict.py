from sklearn.externals import joblib
from keras.models import load_model
from tools import proc_text,cal_bow
from keras.preprocessing import sequence
import numpy as np

def NB_predict(string):
    nb_model = joblib.load('NaiveBayes_2000.pkl')
    bow = cal_bow(string)
    answer = nb_model.predict([bow])
    return answer
    # print('贝叶斯分类器对字符串：{0} 的预测标签为 {1}'.format(string, answer))


def CNN_predict(string):
    model = load_model('text_cnn_finally.h5')
    tokenizer = joblib.load('tokenizer.pkl')   
    pro = proc_text(string)
    seq = tokenizer.texts_to_sequences([pro])
    vec = sequence.pad_sequences(seq, maxlen=100)
    answer = model.predict(vec)
    label = (1 if (answer[0] > 0.5) else 0)
    return label

def LSTM_predict(string):
    model = load_model('lstm_first.h5')
    tokenizer = joblib.load('tokenizer.pkl')
    pro = proc_text(string)
    seq = tokenizer.texts_to_sequences([pro])
    vec = sequence.pad_sequences(seq, maxlen=100)
    answer = model.predict(vec)
    label = (1 if (answer[0] > 0.5) else 0)
    return label
