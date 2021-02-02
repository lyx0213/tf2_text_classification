#-*- coding:utf-8 -*-
# @Time:2021/1/25 14:45
# @File：TextRNN2.py


import tensorflow as tf


class TextRNN2(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size, sentence_len, dropout_ratio, label_nums, allow_cudnn_kernel=False, **kwargs):
        super(TextRNN2, self).__init__()
        print('{} model loaded ' .format(self.__class__.__name__))
        self.hidden_size = embedding_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size,
                                                   input_length=sentence_len, name='embedding_layer')

        if allow_cudnn_kernel:
            if dropout_ratio:
                self.bilstm = tf.keras.layers.Bidirectional(tf.nn.RNNCellDropoutWrapper(
                    tf.keras.layers.LSTM(units=self.hidden_size, return_sequences=True),output_keep_prob=dropout_ratio))
            else:
                self.bilstm = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(units=self.hidden_size, return_sequences=True))

            self.lstm = tf.keras.layers.LSTM(units=self.hidden_size*2,)
            self.dropout = tf.keras.layers.Dropout(rate=dropout_ratio)
            self.linear1 = tf.keras.layers.Dense(units=self.hidden_size*2, activation='tanh')
            self.linear = tf.keras.layers.Dense(units=label_nums, activation='softmax')
        else:
            self.lstm_cell1 = tf.keras.layers.LSTMCell(units=self.hidden_size, dropout=dropout_ratio)
            self.lstm_cell2 = tf.keras.layers.LSTMCell(units=self.hidden_size * 2, dropout=dropout_ratio)
            self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(self.lstm_cell1, return_sequences=True))
            self.lstm = tf.keras.layers.RNN(self.lstm_cell2)
            self.dropout = tf.keras.layers.Dropout(rate=dropout_ratio)
            self.linear1 = tf.keras.layers.Dense(units=self.hidden_size * 2, activation='tanh')
            self.linear2 = tf.keras.layers.Dense(units=label_nums, activation='softmax')


    def call(self, inputs, training=None, mask=None):
        embedded = self.embedding(inputs)
        bilstm_output = self.bilstm(embedded)       #shape=(batch_size, hidde_size*2)
        bilstm_output = self.dropout(bilstm_output)
        lstm_output = self.lstm(bilstm_output)
        output = self.dropout(lstm_output)
        output = self.linear1(output)
        output = self.linear2(output)
        return output
