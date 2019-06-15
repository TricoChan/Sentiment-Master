# coding=gbk
import datetime
from tools import proc_text,extract_feat_from_string
from sklearn.externals import joblib
from nltk import Text
import numpy as np
from predict import CNN_predict,LSTM_predict


# 时间差计算函数
def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1

def run_main():
    # --------------------------------------时间计算---------------------------------
    '''
    startdate = datetime.datetime.now()
    # 当前时间转换为指定字符串格
    startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")
    enddate = datetime.datetime.now()
    enddate = enddate.strftime("%Y-%m-%d %H:%M:%S")
    print(subtime(startdate,enddate))
    # np.array
    '''
    #CNN_predict('首先外观做工，索尼大法好就完事儿，做工扎实，镀金的SONY出街#格满满，黑色沉稳大气，五星。音质，相比竞品~C35，闭着眼睛就能秒杀BOSE，音质比35好太多了，这也是选择索尼这款的原因之一，另外就算耳机没电了也可以连接线材继续听歌，音质也提升很多，这点很好。连接蓝牙速度超快，操作触控面板很黑科技，非常实用，而且有自动调节场景功能，真的良心。掩着一只就可以听环境音不要太棒帮哦！缺点:相比~C35佩戴不是很舒适，时间久了有点压耳朵。')
    # CNN_predict('很不开心的一次购物')
    # CNN_predict('好开心啊')
    # CNN_predict('真的不错')
    # CNN_predict('我喜欢')
    # CNN_predict('真恶心')
    # CNN_predict('消极')
    # CNN_predict('积极')
    # CNN_predict('不是吧')
    # CNN_predict('真的是够了')
    # LSTM_predict('不喜欢这个电影')





if __name__ == '__main__':
    run_main()