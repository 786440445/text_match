import os
import sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import re
import jieba
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from undergraduate import args
from utils.data_utils import shuffle, pad_sequences
import math

jieba.load_userdict(os.path.join(home_dir, "data/tx_dict/500000-dict.txt"))
vocab_path = os.path.join(home_dir, 'data/vocab.txt')
words_vocab_path = os.path.join(home_dir, 'data/vocab_words.txt')

# 参数集合
corpus_path = os.path.join(home_dir, 'data/clean_lcqmc')


class dataloader():
    def __init__(self, file, batch_size):
        # 加载char_index、静态词向量、动态词向量的训练数据
        path = os.path.join(corpus_path, file)
        df = pd.read_csv(path, header=None, sep='\t')
        p = df.iloc[:, 0].values
        h = df.iloc[:, 1].values
        label = df.iloc[:, 2].values
        self.batch_size = batch_size
        if args.shuffle:
            self.p, self.h, self.label = shuffle(p, h, label)
        else:
            self.p, self.h, self.label = p, h, label
        self.data_length = len(self.p)
        self.nums_batchs = math.floor(self.data_length / self.batch_size)

    def generator(self):
        for i in range(len(self)):
            p_c_index, h_c_index, p_w_index, h_w_index, same_word, label_batch = self.__getitem__(i)
            yield p_c_index, h_c_index, p_w_index, h_w_index, same_word, label_batch

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.data_length)
        batch_indexs = list(range(start_index, end_index))
        # 根据索引获取datas集合中的数据
        p_batch = [self.p[k] for k in batch_indexs]
        h_batch = [self.h[k] for k in batch_indexs]
        label_batch = [self.label[k] for k in batch_indexs]
        p_c_index, h_c_index = char_index(p_batch, h_batch)

        p_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…///-/s]", "", x))), p_batch))
        h_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…///-/s]", "", x))), h_batch))
        p_w_index, h_w_index = word_index(p_seg, h_seg)
        same_word = []
        for p_i, h_i in zip(p_w_index, h_w_index):
            dic = {}
            for i in p_i:
                if i == 0:
                    break
                dic[i] = dic.get(i, 0) + 1
            for index, i in enumerate(h_i):
                if i == 0:
                    same_word.append(0)
                    break
                dic[i] = dic.get(i, 0) - 1
                if dic[i] == 0:
                    same_word.append(1)
                    break
                if index == len(h_i) - 1:
                    same_word.append(0)

        return p_c_index, h_c_index, p_w_index, h_w_index, same_word, label_batch

    def __len__(self):
        return self.nums_batchs

# 加载字典
def load_char_vocab():
    vocab = [line.strip() for line in open(vocab_path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


# 加载词典
def load_word_vocab():
    vocab = [line.strip() for line in open(words_vocab_path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word



# 字->index
def char_index(p_sentences, h_sentences):
    word2idx, idx2word = load_char_vocab()
    p_list, h_list = [], []

    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=args.max_char_len)
    h_list = pad_sequences(h_list, maxlen=args.max_char_len)

    return p_list, h_list


# 词->index
def word_index(p_sentences, h_sentences):
    word2idx, idx2word = load_word_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=args.max_word_len)
    h_list = pad_sequences(h_list, maxlen=args.max_word_len)

    return p_list, h_list