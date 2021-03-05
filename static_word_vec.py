#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：nlp_tools -> static_word_vec
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/11/18 8:36 PM
@Desc   ：
=================================================='''
import os, sys
import numpy as np
from tqdm import tqdm
home_dir = os.getcwd()
sys.path.append(home_dir)

from utils.load_data import load_char_vocab
from gensim.models import Word2Vec, word2vec, KeyedVectors
import pandas as pd
import jieba
#
def loadEmbedding(embeddingFile, word2id, embeddingSize=200):
    """ Initialize embeddings with pre-trained word2vec vectors
    Will modify the embedding weights of the current loaded model
    sess：会话
    embeddingFile：Tencent_AILab_ChineseEmbedding.txt的路径
    word2id：自己数据集中的word2id
    embeddingSize: 词向量的维度，我这里直接设置的200，和原始一样，低于200的采用我屏蔽掉的代码应该可以，我还没测

    """
    print("Loading pre-trained word embeddings from %s " % embeddingFile)
    with open(embeddingFile, "r", encoding='ISO-8859-1') as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        initW = np.random.uniform(-0.25, 0.25, (len(word2id), vector_size))
        for i in tqdm(range(vocab_size)):
            line = f.readline()
            lists = line.split(' ')
            word = lists[0]
            if word in word2id:
                number = map(float, lists[1:])
                number = list(number)
                vector = np.array(number)
                initW[word2id[word]] = vector
    return initW

df = pd.read_csv(os.path.join(home_dir, 'data/clean_lcqmc/train.txt'), header=None, sep='\t')
p = df.iloc[:, 0].values
h = df.iloc[:, 1].values
p_seg = list(map(lambda x: list(jieba.cut(x)), p))
h_seg = list(map(lambda x: list(jieba.cut(x)), h))
common_texts = []
common_texts.extend(p_seg)
common_texts.extend(h_seg)

df = pd.read_csv(os.path.join(home_dir, 'data/clean_lcqmc/test.txt'), header=None, sep='\t')
p = df.iloc[:, 0].values
h = df.iloc[:, 1].values
p_seg = list(map(lambda x: list(jieba.cut(x)), p))
h_seg = list(map(lambda x: list(jieba.cut(x)), h))
common_texts.extend(p_seg)
common_texts.extend(h_seg)

df = pd.read_csv(os.path.join(home_dir, 'data/clean_lcqmc/dev.txt'), header=None, sep='\t')
p = df.iloc[:, 0].values
h = df.iloc[:, 1].values
p_seg = list(map(lambda x: list(jieba.cut(x)), p))
h_seg = list(map(lambda x: list(jieba.cut(x)), h))
common_texts.extend(p_seg)
common_texts.extend(h_seg)

# embeding_path = os.path.join(home_dir, "text_match/tx_embedding/500000-small.txt")
# word2idx, idx2word = load_char_vocab()
# embedding_table = loadEmbedding(embeding_path, word2id=word2idx)

# wv_from_text = KeyedVectors.load_word2vec_format(embeding_path, limit=4000000, binary=False)
# # 使用init_sims会比较省内存
# wv_from_text.init_sims(replace=True)
# # 重新保存加载变量为二进制形式
# bin_path = os.path.join(home_dir, "tx_embedding/embedding.bin")
# print(save_path)
# wv_from_text.save(save_path)
# model = Word2Vec.load(embedding_table)
print('success')
# model.init_sims(replace=True)
model = Word2Vec(common_texts, size=200, window=5, min_count=5, workers=12)
print('文本长度', len(common_texts))
model.save(os.path.join(home_dir, "output/word2vec/word2vec.model"))

