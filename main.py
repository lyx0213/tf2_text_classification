#-*- coding:utf-8 -*-
# @Time:2021/1/19 16:46
# @Auther :lizhe
# @File：main.py
# @Email:bylz0213@gmail.com


import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf

from tqdm import tqdm
# from tensorflow.keras.models import Model
from train_test_step import TrainTestStep
from models.FastText import FastText
from models.TextCNN import TextCNN
from models.TextRNN import TextRNN
from models.TextRNN2 import TextRNN2
from data_process.data_process import DataProcess
from predict_label import model_predict

FLAGS = tf.compat.v1.app.flags.FLAGS
tf.compat.v1.app.flags.DEFINE_integer("embedding_size", 100, "words embedding size")
tf.compat.v1.app.flags.DEFINE_integer("epochs", 10, "iterations")
tf.compat.v1.app.flags.DEFINE_integer("batch_size", 512, "batch size")
tf.compat.v1.app.flags.DEFINE_integer("sentence_len", 15, "sentence length")
tf.compat.v1.app.flags.DEFINE_float("dropout_ratio",0.5, "dropout ratio")
tf.compat.v1.app.flags.DEFINE_integer("hidden_size", 100, "dnn hidden size")
tf.compat.v1.app.flags.DEFINE_integer("filter_nums", 150, "cnn kernel nums")
tf.compat.v1.app.flags.DEFINE_string("filter_sizes", "3,4,5", "cnn kernel size")  # separated by commas
tf.compat.v1.app.flags.DEFINE_integer("cnn_strides", 1, "cnn kernel strides")
tf.compat.v1.app.flags.DEFINE_integer("model_loss_init", 10000, "init save model min loss")
tf.compat.v1.app.flags.DEFINE_string("model", "FastText", "model")  # The optional models are in the models directory
tf.compat.v1.app.flags.DEFINE_boolean("use_jieba", True, "use jieba seg words")
tf.compat.v1.app.flags.DEFINE_boolean("is_eval", False, "eval verification")
tf.compat.v1.app.flags.DEFINE_boolean("is_test", True, "test verification")
tf.compat.v1.app.flags.DEFINE_boolean("is_predict", True, "predict result")
tf.compat.v1.app.flags.DEFINE_boolean("data_format", False, "default sentence\tlabel\n if False label\tsentence\n")
tf.compat.v1.app.flags.DEFINE_string("train_file_path", "./datas/train.tsv", "train file path")
tf.compat.v1.app.flags.DEFINE_string("eval_file_path", "./datas/val.tsv", "eval file path")
tf.compat.v1.app.flags.DEFINE_string("test_file_path", "./datas/test.tsv", "test file path")
tf.compat.v1.app.flags.DEFINE_string("train_seg_file_path", "./datas/seg_results/train.seg", "train seg file path")
tf.compat.v1.app.flags.DEFINE_string("eval_seg_file_path", "./datas/seg_results/eval.seg", "eval seg file path")
tf.compat.v1.app.flags.DEFINE_string("test_seg_file_path", "./datas/seg_results/test.seg", "test seg file path")
tf.compat.v1.app.flags.DEFINE_string("label_vocab_path", "./trained_vocab_result/dict.pkl", "")




def main(_):

    data_process = DataProcess(FLAGS.batch_size, FLAGS.use_jieba, FLAGS.data_format)
    data_process.data_seg(FLAGS.train_file_path, FLAGS.train_seg_file_path, FLAGS.label_vocab_path)
    train_x, train_y = data_process.word2int(FLAGS.train_seg_file_path)

    if FLAGS.is_eval:
        data_process.data_seg(FLAGS.eval_file_path, FLAGS.eval_seg_file_path, FLAGS.label_vocab_path, module='eval')
        eval_x, eval_y = data_process.word2int(FLAGS.eval_seg_file_path)

    if FLAGS.is_test:
        data_process.data_seg(FLAGS.test_file_path, FLAGS.test_seg_file_path, FLAGS.label_vocab_path, module='test')
        test_x, test_y = data_process.word2int(FLAGS.test_seg_file_path)

    model = eval(FLAGS.model)(
        vocab_size=len(data_process.vocab2int),
        embedding_size=FLAGS.embedding_size,
        sentence_len=FLAGS.sentence_len,
        hidden_size=FLAGS.hidden_size,
        dropout_ratio=FLAGS.dropout_ratio,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
        filter_nums=FLAGS.filter_nums,
        cnn_strides=FLAGS.cnn_strides,
        label_nums=data_process.label_nums)

    train_test_step_ = TrainTestStep(model)
    min_test_loss = FLAGS.model_loss_init
    for epoch in range(FLAGS.epochs):
        COUNTS = 0
        # 在下一个epoch开始时，重置评估指标
        train_test_step_.train_loss.reset_states()
        train_test_step_.train_accuracy.reset_states()
        train_test_step_.eval_loss.reset_states()
        train_test_step_.eval_accuracy.reset_states()
        train_test_step_.test_loss.reset_states()
        train_test_step_.test_accuracy.reset_states()


        for tokens, labels in tqdm(data_process.get_batch(train_x ,train_y)):
            COUNTS += len(labels)
            if COUNTS % 100000 < FLAGS.batch_size:
                print('have loading %d samples'%COUNTS)
            labels = np.reshape(np.array(labels,dtype=int),(-1,1))
            train_test_step_.train_step(tf.keras.preprocessing.sequence.pad_sequences(
                tokens, maxlen=FLAGS.sentence_len, padding='post', truncating='post'), labels)

        if FLAGS.is_eval:
            for eval_tokens, eval_labels in data_process.get_batch(eval_x, eval_y):
                eval_labels = np.reshape(np.array(eval_labels,dtype=int), (-1, 1))
                train_test_step_.eval_step(tf.keras.preprocessing.sequence.pad_sequences(
                    eval_tokens, maxlen=FLAGS.sentence_len, padding='post', truncating='post'), eval_labels)

        if FLAGS.is_test:
            for test_tokens, test_labels in data_process.get_batch(test_x, test_y):
                test_labels = np.reshape(np.array(test_labels,dtype=int), (-1, 1))
                train_test_step_.test_step(tf.keras.preprocessing.sequence.pad_sequences(
                    test_tokens, maxlen=FLAGS.sentence_len, padding='post', truncating='post'), test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {},' \
                   ' Eval Loss: {}, Eval Accuracy: {},' \
                   ' Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_test_step_.train_loss.result(),
                              train_test_step_.train_accuracy.result() * 100,
                              train_test_step_.eval_loss.result(),
                              train_test_step_.eval_accuracy.result() * 100,
                              train_test_step_.test_loss.result(),
                              train_test_step_.test_accuracy.result() * 100))

        if min_test_loss > train_test_step_.test_loss.result():
            min_test_loss = train_test_step_.test_loss.result()
            print('第{}个epoch模型保存'.format(epoch))
            model.save("my_model")

    if FLAGS.is_predict:
        input_ = '10	李克强抵达东京出席第七次中日韩领导人会议并对日本进行正式访问'
        input_ = input_.strip().split('\t')
        label = input_[0]
        tokens = input_[1]
        model_predict(tokens, label, FLAGS.sentence_len, FLAGS.label_vocab_path)





if __name__ == '__main__':

    tf.compat.v1.app.run()




