#-*- coding:utf-8 -*-
# @Time:2021/1/25 18:23
# @Auther :lizhe
# @File：data_process.py
# @Email:bylz0213@gmail.com
import os
import random
import pickle as pkl
from random import shuffle


random.seed(2020)

from jieba2.jieba import JieBa



class CreateVocab:
    def __init__(self):
        self.vocab2int = {}
        self.vocab2int['padding'] = 0
        self.vocab2nums = 1
        self.label_nums = 0
        self.label2int = None

    def data_seg(self, input_file_path, output_file_path, label_vocab_path, module='train'):
        output_file = open(output_file_path, 'w', encoding='utf-8')
        label_list = []
        with open(input_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()[1:]
            shuffle(lines)
            for index,line in enumerate(lines):
                if index == 0:
                    continue
                else:
                    line = line.strip().split('\t')
                    contents = self.jieba_cut.words_seg(line[self.content_index]) if self.use_jieba else \
                        line[self.content_index].strip().split(' ')
                    label = line[self.label_index]
                    content = ''
                    for word in contents:
                        if module == 'train' and '__' + word + '__' not in self.vocab2int:
                            self.vocab2int['__' + word + '__'] = self.vocab2nums
                            self.vocab2nums += 1
                            label_list.append(label)
                        content += '__' + word + '__ '
                    if self.data_format:
                        output_file.write(content[:-1] + '\t' +  + '\n')
                    else:
                        output_file.write(label + '\t' + content[:-1] + '\n')
        if module == 'train':

            label_list = list(set(label_list))
            self.label_nums = len(label_list)
            self.label2int = {label: i for i, label in enumerate(label_list)}
            self.int2label = {value: key for key,value in self.label2int.items()}
            dict = []
            dict.extend([self.vocab2int, self.label2int,self.int2label])
            pkl.dump(dict,open(label_vocab_path,'wb'))


class DataProcess(CreateVocab):

    def __init__(self, batch_size, use_jieba, data_format):
        super(DataProcess, self).__init__()
        self.jieba_cut = JieBa()
        self.batch_size = batch_size
        self.use_jieba = use_jieba
        self.data_format = data_format
        if self.data_format:
            self.content_index = 0
            self.label_index = 1
        else:
            self.content_index = 1
            self.label_index = 0

    def word2int(self, input_file_path):
        self.temp_list = []
        self.temp_label = []

        with open(input_file_path, encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                content = line[self.content_index].split(' ')
                self.temp_list.append([self.vocab2int.get(i, self.vocab2int['padding']) for i in content])
                self.temp_label.append(self.label2int[line[self.label_index]])
        print('{},数据总数为：{}'.format(input_file_path, len(self.temp_label)))

        return self.temp_list, self.temp_label

    def get_batch(self, temp_list, temp_label):
        for i in range(0, int(len(temp_label) / self.batch_size)):
            if (i+1) * self.batch_size > len(temp_label):
                continue
            yield temp_list[i * self.batch_size:(i+1) * self.batch_size],\
                  temp_label[i * self.batch_size:(i+1) * self.batch_size]