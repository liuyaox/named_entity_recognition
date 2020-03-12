# -*- coding: utf-8 -*-
"""
Created:    2019-11-13 17:32:19
Author:     liuyao8
Descritipn: 数据预处理，包括解析数据文件，生成字典，数据编码，数据解码，用于一般场景和BERT
"""
import numpy as np
from collections import Counter
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras_bert import Tokenizer



# 1. 数据预处理，用于一般场景

def data_parsing(data_file):
    """
    解析语料文件：以空行(\n\n)分隔各样本(句子)，各样本内每行是一个字及其标注，以空格分隔
    例如某个样本前5行分别是'中 B-ORG','共 I-ORG','中 I-ORG','央 I-ORG','致 O'，转化为样本列表
    其中每个样本为：[['中', 'B-ORG'], ['共', 'I-ORG'], ['中', 'I-ORG'], ['央', 'I-ORG'], ['致', 'O']]
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
    Y = np.expand_dims(Y, 2)           # (, maxlen) --> (, maxlen, 1)  与CRF输出的shape一致
    return X, Y


def get_word2idx(data, min_freq):
    '''生成字典和映射字典'''
    word_counts = Counter(row[0].lower() for sample in data for row in sample)  # 基于train
    vocab = [w for w, f in word_counts.items() if f >= min_freq]         # 以频率排序为id
    word2idx = {w: i for i, w in enumerate(vocab)}    # TODO 不应该从0和1开始吧？应该从2开始？
    idx2word = {i: w for (w, i) in word2idx.items()}
    return word2idx, idx2word, vocab
    

    
# 2. 数据预处理，用于BERT
    
def preprocess_data(path):
    '''每行一个字一个tag --> 样本列表，样本tags列表'''
    sentences = []
    tags = []
    with open(path, encoding="utf-8") as data_file:
        for sentence in data_file.read().strip().split('\n\n'):
            _sentence = ""
            tag = []
            for word in sentence.strip().split('\n'):
                content = word.strip().split()
                _sentence += content[0]
                tag.append(content[1])
            sentences.append(_sentence)
            tags.append(tag)
    return sentences, tags



class DataProcessForBert:

    def __init__(self, config, seq_maxlen=None, bert_vocab_file=None):
        self.config = config
        self.seq_maxlen = seq_maxlen if seq_maxlen else config.SEQ_MAXLEN
        bert_vocab_file = bert_vocab_file if bert_vocab_file else config.bert_vocab_file
        self.load_dict(bert_vocab_file)


    def load_dict(self, bert_vocab_file):
        '''生成Tag字典和BERT字典'''
        # Tag字典
        #           {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6}
        # {'PAD': 0, 'O': 1, 'B-PER': 2, 'I-PER': 3, 'B-LOC': 4, 'I-LOC': 5, 'B-ORG': 6, 'I-ORG': 7}
        self.tag2idx = {tag: i for i, tag in enumerate(self.config.TAGS)}
        self.idx2tag = {v: k for (k, v) in self.tag2idx.items()}
        # BERT字典
        self.vocab = {}
        with open(bert_vocab_file, 'r+', encoding='utf-8') as f_vocab:
            for line in f_vocab:
                self.vocab[line.strip()] = len(self.vocab)
        # TODO 不需要从1或2开始，因为vocab.txt中已经有[PAD],[UNK],[CLS],[SEP],[MASK],<S>,<T>这些！尤其[PAD]，就是第1个

        
    def encode_input_x(self, sentences):
        '''数据X序列化编码  使用BERT的Tokenizer：Token编码, 句子编码   sentences是句子列表，字符串'''
        tokenizer = Tokenizer(self.vocab)
        sent_token_ids = []
        sent_segment_ids = []
        for sequence in sentences:
            token_ids, segment_ids = tokenizer.encode(first=sequence, max_len=self.seq_maxlen)  # 输入只有1个句子！
            sent_token_ids.append(token_ids)
            sent_segment_ids.append(segment_ids)
        return [sent_token_ids, sent_segment_ids]

        
    def encode_input_y(self, tags):
        '''数据Y序列化编码  tags是tag列表(对应一个sentence)的列表'''
        tag_ids = []
        for line in tags:
            # TODO 第1个和最后1个0是针对[CLS]和[SEP] ?
            tag = [0] + [int(self.tag2idx[tag.strip()]) for tag in line] + [0]
            tag_ids.append(tag)

        # TODO pad id也是0？与O这个tag一样的id？
        pad_tags = pad_sequences(tag_ids, maxlen=self.seq_maxlen, padding='post', truncating='post')
        tag_ids = np.expand_dims(pad_tags, 2)
        return tag_ids


    def decode_output(self, tag_probs):
        '''模型输出结果取最大概率的tag_id并解码为tag'''
        tag_ids = [np.argmax(prob) for prob in tag_probs]
        return [self.idx2tag[int(x)] for x in tag_ids]
    
  
  
# 3. 数据处理

def data_config_processing(config):
    """准备好data和config"""
    train = data_parsing(config.train_file)
    test = data_parsing(config.test_file)
    config.MAXLEN = max(len(sample) for sample in train)   # 100
    
    word2idx, idx2word, vocab = get_word2idx(train, config.MIN_FREQ)
    config.word2idx = word2idx
    config.VOCABSIZE = len(vocab)
    
    x_train, y_train = data_encoding(train, word2idx, config.TAGS, config.MAXLEN)
    x_test, y_test = data_encoding(test, word2idx, config.TAGS, config.MAXLEN)
    
    pickle.dump((x_train, y_train, x_test, y_test), open(config.data_encoded_file, 'wb'))
    pickle.dump(config, open(config.config_file, 'wb'))
  
  
def data_config_processing_bert(config):
    '''准备好data和config，for BERT'''
    dp = DataProcessForBert(config)
    
    train_sents, train_tags = preprocess_data(config.train_file)
    test_sents, test_tags = preprocess_data(config.test_file)
    x_train, y_train = dp.encode_input_x(train_sents), dp.encode_input_y(train_tags)
    x_test, y_test = dp.encode_input_x(test_sents), dp.encode_input_y(test_tags)
  
    pickle.dump((x_train, y_train, x_test, y_test), open(config.data_encoded_bert_file, 'wb'))
    
    
    
if __name__ == '__main__':
    
    from Config import Config
    config = Config()
    
    data_config_processing(config)
    
    config.set_pad_same_with_o(False)
    data_config_processing_bert(config)
    