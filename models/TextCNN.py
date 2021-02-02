#-*- coding:utf-8 -*-
# @Time:2021/1/23 9:52
# @File：TextCNN.py
import tensorflow as tf


class ConvBlock(tf.keras.Model):

    def __init__(self, embedding_size, sentence_len, filter_nums, filter_size, cnn_strides, block_index, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filter_nums, kernel_size=[filter_size, embedding_size],
                                                     strides=cnn_strides, activation='relu', name='c' + str(block_index))

        self.bn = tf.keras.layers.BatchNormalization(trainable=True, name='bn' + str(block_index))
        self.pool = tf.keras.layers.MaxPool2D(pool_size=[sentence_len - filter_size + 1, 1], strides=cnn_strides,
                                                            padding='VALID', name='p' + str(block_index))

    def call(self, inputs, training=None, mask=None):

        output_result = self.conv(inputs)
        output_result = self.bn(output_result)
        output_result = self.pool(output_result)
        return output_result


class TextCNN(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size, filter_sizes, filter_nums,
                 cnn_strides, sentence_len, dropout_ratio, label_nums, **kwargs):
        super(TextCNN, self).__init__()
        print('{} model loaded ' .format(self.__class__.__name__))
        self.filter_nums = filter_nums
        self.filter_sizes = filter_sizes
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=sentence_len)
        self.module_list = []
        for i, filter_size in enumerate(self.filter_sizes):
            self.module_list.append(ConvBlock(embedding_size, sentence_len, filter_nums, filter_size, cnn_strides, i))
        self.dropout = tf.keras.layers.Dropout(dropout_ratio)
        self.linear = tf.keras.layers.Dense(label_nums, activation='tanh', use_bias=True)

    def call(self, inputs, training=None, mask=None):

        input_emb = self.embedding(inputs)  #shape=(512, 15, 100)
        input_emb = tf.expand_dims(input_emb, -1)
        output_result = []
        for i in self.module_list:
            output_result.append(i(input_emb))
        output_result = tf.keras.layers.concatenate(output_result)
        output_result = tf.reshape(output_result, (-1, self.filter_nums*3))
        output_result = self.dropout(output_result)
        output_result = self.linear(output_result)

        return output_result








