# coding=gbk
import datetime
from tools import proc_text,extract_feat_from_string
from sklearn.externals import joblib
from nltk import Text
import numpy as np
from predict import CNN_predict,LSTM_predict


# ʱ�����㺯��
def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1

def run_main():
    # --------------------------------------ʱ�����---------------------------------
    '''
    startdate = datetime.datetime.now()
    # ��ǰʱ��ת��Ϊָ���ַ�����
    startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")
    enddate = datetime.datetime.now()
    enddate = enddate.strftime("%Y-%m-%d %H:%M:%S")
    print(subtime(startdate,enddate))
    # np.array
    '''
    #CNN_predict('�����������������󷨺þ����¶���������ʵ���ƽ��SONY����#����������ɫ���ȴ��������ǡ����ʣ���Ⱦ�Ʒ~C35�������۾�������ɱBOSE�����ʱ�35��̫���ˣ���Ҳ��ѡ����������ԭ��֮һ������������û����Ҳ���������߲ļ������裬����Ҳ�����ܶ࣬���ܺá����������ٶȳ��죬�����������ܺڿƼ����ǳ�ʵ�ã��������Զ����ڳ������ܣ�������ġ�����һֻ�Ϳ�������������Ҫ̫����Ŷ��ȱ��:���~C35������Ǻ����ʣ�ʱ������е�ѹ���䡣')
    # CNN_predict('�ܲ����ĵ�һ�ι���')
    # CNN_predict('�ÿ��İ�')
    # CNN_predict('��Ĳ���')
    # CNN_predict('��ϲ��')
    # CNN_predict('�����')
    # CNN_predict('����')
    # CNN_predict('����')
    # CNN_predict('���ǰ�')
    # CNN_predict('����ǹ���')
    # LSTM_predict('��ϲ�������Ӱ')





if __name__ == '__main__':
    run_main()