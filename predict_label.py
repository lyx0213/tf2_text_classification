#-*- coding:utf-8 -*-
# @Time:2021/1/26 18:43
# @File：predict_label.py
import pickle
import numpy as np
import tensorflow as tf
from jieba2.jieba import JieBa

def model_predict(predict_sentence, label, sentence_len, word2int_path):

    with open(word2int_path, 'rb') as f:
        vocab2int, label2int, int2label = pickle.load(f)

    jieba_cut = JieBa()
    print(predict_sentence)
    a_seg = [vocab2int.get('__' + i + '__', vocab2int['padding'])
             for i in jieba_cut.words_seg(predict_sentence)][:sentence_len]
    print('分词后的结果：', a_seg)

    a = np.array(a_seg).reshape(1, -1).astype(np.int32)

    if a.size < sentence_len:
        a = np.hstack((a, np.zeros((1, sentence_len - a.size), dtype=np.int32)))

    tensorflow_graph = tf.saved_model.load("my_model")
    predicted = tensorflow_graph(a).numpy()

    print('预测的label为：{},原始lable为{}'.format(int2label[predicted.argmax()], label))



