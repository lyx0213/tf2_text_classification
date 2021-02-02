#-*- coding:utf-8 -*-
# @Time:2021/1/25 18:48
# @File：train_test_step.py
import tensorflow as tf


class TrainTestStep:
    def __init__(self, model):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.eval_loss = tf.keras.metrics.Mean(name='train_loss')
        self.eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.model = model


    @tf.function
    def train_step(self,tokens, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(tokens, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self,tokens, labels):
        predictions = self.model(tokens)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    @tf.function
    def eval_step(self,tokens, labels):
        predictions = self.model(tokens)
        e_loss = self.loss_object(labels, predictions)

        self.eval_loss(e_loss)
        self.eval_accuracy(labels, predictions)

