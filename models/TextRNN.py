#-*- coding:utf-8 -*-
# @Time:2021/1/25 13:51
# @Auther :lizhe
# @File：TextRNN.py
# @Email:bylz0213@gmail.com


import tensorflow as tf


class TextRNN(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size, sentence_len, dropout_ratio, label_nums,
                 allow_cudnn_kernel=False, **kwargs):
        super(TextRNN, self).__init__()
        print('{} model loaded ' .format(self.__class__.__name__))
        self.hidden_size = embedding_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size,
                                                   input_length=sentence_len, name='embedding_layer')

        if allow_cudnn_kernel:
            if dropout_ratio:
                self.bilstm = tf.keras.layers.Bidirectional(tf.nn.RNNCellDropoutWrapper(
                    tf.keras.layers.LSTM(units=self.hidden_size), output_keep_prob=dropout_ratio))
                self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.hidden_size))
        else:
            self.lstm_cell = tf.keras.layers.LSTMCell(units=self.hidden_size)
            # if dropout_ratio:
            #     self.lstm_cell = tf.nn.RNNCellDropoutWrapper(tf.keras.layers.LSTMCell(units=self.hidden_size),
            #                                                  output_keep_prob=dropout_ratio)

            self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(self.lstm_cell))

        self.dropout = tf.keras.layers.Dropout(rate=dropout_ratio)
        self.linear = tf.keras.layers.Dense(units=label_nums, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        embedded = self.embedding(inputs)
        bilstm_output = self.bilstm(embedded)
        output = self.dropout(bilstm_output)
        output = self.linear(output)
        return output


