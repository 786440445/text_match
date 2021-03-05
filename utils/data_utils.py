import sys, os
home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(home_dir)

import numpy as np
import pandas as pd
from collections import Counter

import jieba

jieba.load_userdict(os.path.join(home_dir, "data/tx_dict/500000-dict.txt"))

def analysis_vocab_length():
    len_dict = {}
    all_count = 0
    path = os.path.join(home_dir, 'data/lcqmc')
    for path, _ , files in os.walk(path):
        for file in files:
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    line = line.split('\t')
                    q = line[0]
                    a = line[1]
                    all_count += 2
                    len_dict[len(q)] = len_dict.get(len(q), 0) + 1
                    len_dict[len(a)] = len_dict.get(len(a), 0) + 1

    len_dict = sorted(len_dict.items(), key=lambda x:x[0])
    sums = 0
    print(len_dict)
    for leng, count in len_dict:
        sums += count
        if leng % 5 == 0:
            print(leng, '\t', round(sums / all_count, 4) * 100, '%')


def analysis_posi_neg():
    path = os.path.join(home_dir, 'data/lcqmc')
    for path, _ , files in os.walk(path):
        for file in files:
            posi = 0
            neg = 0
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    line = line.split('\t')
                    label = line[2]
                    if int(label) == 1:
                        posi += 1
                    else:
                        neg += 1
            print(file, 'posi_nums: ', posi, 'neg_nums: ', neg)


# 构建字表
def build_vocab():
    chars = ''
    path = os.path.join(home_dir, 'data/clean_lcqmc')
    for path, _, files in os.walk(path):
        for file in files:
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line != '':
                        line = line.split('\t')
                        s1 = line[0]
                        s2 = line[1]
                        chars += s1 + s2

    dic = Counter(list(chars))
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)

    chars = [item[0] for item in dic if item[1] >= 4]
    chars = [item for item in chars if u'\u4e00' <= item <= u'\u9fa5' or item in ['!', '?']]
    chars = ['<PAD>', '<UNK>'] + chars
    with open(os.path.join(home_dir, 'data/vocab.txt'), 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(chars))

# 构建词表
def build_words_vocab():
    chars = []
    path = os.path.join(home_dir, 'data/clean_lcqmc')
    for path, _, files in os.walk(path):
        for file in files:
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line != '':
                        line = line.split('\t')
                        s1 = line[0]
                        s2 = line[1]
                        s1 = jieba.lcut(s1)
                        for item in s1:
                            if len(item) > 1:
                                chars.append(item)
                        s2 = jieba.lcut(s2)
                        for item in s2:
                            if len(item) > 1:
                                chars.append(item)
    dic = Counter(chars)
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)

    chars = [item[0] for item in dic if item[1] >= 4]
    chars = ['</s>'] + chars
    with open(os.path.join(home_dir, 'data/vocab_words.txt'), 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(chars))


def one_hot(y, nb_classes):
    """ one_hot

    向量转one-hot

    Arguments:
        y: 带转换的向量
        nb_classes: int 类别数

    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    Y[np.arange(len(y)), y] = 1.
    return Y


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences

    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。

    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值

    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)

    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def shuffle(*arrs):
    """ shuffle

    Shuffle 数据

    Arguments:
        *arrs: 数组数据

    Returns:
        shuffle后的数据

    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)

def calc_f1_score(preds, labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            TP += 1
        elif pred == 1 and label == 0:
            FP += 1
        elif pred == 0 and label == 1:
            TN += 1
        elif pred == 0 and label == 0:
            FN += 1
        else:
            pass
    pre = TP / (TP + FP)
    rec = TP / (TP + TN)
    f1 = 2 * pre * rec / (pre + rec)
    return f1

if __name__ == '__main__':
    # analysis_vocab_length()
    # build_vocab()
    # analysis_posi_neg()
    build_words_vocab()
