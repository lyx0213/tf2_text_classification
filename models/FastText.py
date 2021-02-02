#-*- coding:utf-8 -*-
# @Time:2021/1/21 11:23
# @Auther :lizhe
# @File：FastText.py
# @Email:bylz0213@gmail.com

import tensorflow as tf

class FastText(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, sentence_len, hidden_size, label_nums, model_name='FastText', **kwargs):
        super(FastText, self).__init__()
        print('{} model loaded ' .format(self.__class__.__name__))
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=sentence_len)
        # self.hidden = tf.keras.layers.Dense(hidden_size)
        self.softmax = tf.keras.layers.Dense(label_nums, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        embedding = self.embedding(inputs)
        inputs_x = tf.reduce_mean(embedding, axis=1)
        result = self.softmax(inputs_x)
        return result






