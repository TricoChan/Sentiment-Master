# Sentiment-Master
基于深度学习模型的文本情感分析WSGI应用
```java
  class Tihar {
    site : "http://tihar-tech.cn"
  }
```
项目地址
```java
    site : "http://47.101.202.111:9000"
```
## 文件目录
* sentiment
    * /static `前端资源静态文件`
    * /templates `HTML页面`
    * /dataset `数据集`
    * app.py `Python WSGI应用逻辑`
    * data_process.py `朴素贝叶斯`
    * tools.py `分词、去停用词、文本表示构建、训练集划分`
    * ~.char `word2vec预训练词向量`
    * ~.h5 `神经网络模型`
    * ~.pkl `保存的集合文件`
    * ~.npy `保存的矩阵文件`

## 数据集说明
* raw_weibo `原始数据文件`
* clean `数据清洗过后的数据文件`
* delete `加入繁简转换、词性标注预处理的数据文件`
* 关于word2vec `w2v_cnn中，需要根据代码手动查表构建30+GB的文本矩阵表示`

## 问题与交流
* 邮件(ylalppq@126.com)

## 感谢
* [ouxu](https://www.outxu.cn/)
* [shenshen-hungry](https://github.com/Embedding/Chinese-Word-Vectors)
