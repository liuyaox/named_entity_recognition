# -*- coding: utf-8 -*-
"""
Created:    2019-11-13 17:32:19
Author:     liuyao8
Descritipn: 
"""

from numpy import expand_dims
from collections import Counter
import pickle
from keras.preprocessing.sequence import pad_sequences



def data_parsing(data_file):
    """
    解析语料文件：以空行(\n\n)分隔各样本(句子)，各样本内每行是一个字及其标注，以空格分隔
    例如某个样本前5行分别是'中 B-ORG','共 I-ORG','中 I-ORG','央 I-ORG','致 O'
    """
    samples = open(data_file, 'r', encoding='utf8').read().strip().split('\n\n')
    data = [[row.split() for row in sample.split('\n')] for sample in samples]
    return data


def data_encoding(data, word2idx, tags, maxlen):
    """数据编码  Y直接使用indices而非one-hot"""
    X = [[word2idx.get(str(w[0]).lower(), 1) for w in sample] for sample in data]   # TODO UNK为1？
    X = pad_sequences(X, maxlen)
    Y = [[tags.index(w[1]) + 1 for w in sample] for sample in data]     # indices从1开始，以防与mask_value=0冲突？
    Y = pad_sequences(Y, maxlen)    # X与Y的mask_value应该一致，因为貌似源代码里处理Y时使用的是处理X的那个mask
    Y = expand_dims(Y, 2)           # (, maxlen) --> (, maxlen, 1)  与CRF输出的shape一致
    return X, Y


def get_word2idx(data):
    '''生成字典和映射字典'''
    word_counts = Counter(row[0].lower() for sample in data for row in sample)  # 基于train
    vocab = [w for w, f in word_counts.items() if f >= config.MIN_FREQ]         # 以频率排序为id
    word2idx = {w: i for i, w in enumerate(vocab)}    # TODO 不应该从0和1开始吧？应该从2开始？
    idx2word = {i: w for (w, i) in word2idx.items()}
    return word2idx, idx2word, vocab
    

def data_config_processing():
    """准备好data和config"""
    train = data_parsing(config.train_file)
    test = data_parsing(config.test_file)
    config.MAXLEN = max(len(sample) for sample in train)   # 100
    
    word2idx, idx2word, vocab = get_word2idx(train)
    config.word2idx = word2idx
    config.VOCABSIZE = len(vocab)
    
    x_train, y_train = data_encoding(train, word2idx, config.TAGS, config.MAXLEN)
    x_test, y_test = data_encoding(test, word2idx, config.TAGS, config.MAXLEN)
    
    pickle.dump((x_train, y_train, x_test, y_test), open(config.data_encoded_file, 'wb'))
    pickle.dump(config, open(config.config_file, 'wb'))
    
    
  
if __name__ == '__main__':
    
    from Config import Config
    config = Config()
    data_config_processing()
    