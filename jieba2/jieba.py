#-*- coding:utf-8 -*-
# @Time:2021/1/25 18:24
# @Auther :lizhe
# @File：jieba.py
# @Email:bylz0213@gmail.com
import jieba

class JieBa:

    def __init__(self, dict_self=None):
        # jieba.enable_parallel(5)
        jieba.dt.tmp_dir = './jieba2/'
        jieba.dt.cache_file = 'jieba.cache'

        if dict_self:
            jieba.load_userdict(dict_self)

    def words_seg(self,words):
        return jieba.cut(words)