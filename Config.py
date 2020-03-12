# -*- coding: utf-8 -*-
"""
Created:    2019-11-16 13:38:23
Author:     liuyao8
Descritipn: 
"""
import argparse


# 设置多GPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


class Config(object):

    def __init__(self, pad_same_with_o=True):
        
        # Basic
        self.token_level = 'char'           # word: word粒度  char: char粒度  both: word+char粒度
        self.train_file = './data/train_data.txt'
        self.test_file = './data/test_data.txt'
        self.set_pad_same_with_o(pad_same_with_o)
        
        # Data Preprocessing
        self.SEQ_MAXLEN = 80
        self.data_encoded_file = './local/data_encoded.pkl'             # 向量化编码后的训练数据
        
        
        # Embedding
        self.VOCAB_SIZE = 10000
        self.PUBLIC_EMBED_DIM = 200     # 公开训练好的Embedding向量维度
        self.WORD_EMBED_DIM = 150
        self.CHAR_EMBED_DIM = 150
        self.model_word2vec_file = './local/model_word2vec.w2v'         # 训练好的Word Embedding  
        self.model_char2vec_file = './local/model_char2vec.w2v'         # 训练好的Char Embedding
        
        
        # Bert
        self.bert_path = '/home/liuyao58/data/BERT/chinese_L-12_H-768_A-12/'
        self.bert_vocab_file = self.bert_path + 'vocab.txt'
        self.BERT_DIM = 768             # ?
        self.BERT_RNN_UNITS = 64
        self.bert_graph_tmpfile = './tmp_graph_xxx' # ?
        
        
        # Vocabulary
        self.MIN_FREQ = 2
        self.PAD_IDX = 0   # PAD约定取0，不要改变，以下UNK,SOS,EOS可以改变
        self.UNK_IDX = 1   # unknow word   # TODO 原本是没有UNK的？
        self.SOS_IDX = 2   # Start of sentence
        self.EOS_IDX = 3   # End of sentence 
        self.vocab_file = './local/vocab.pkl'       # 词汇表，包含word/char,idx,vector三者之间映射字典，Embedding Layer初始化权重
        
        
        # Model
        self.RNN_UNITS = 200
        self.CNN_UNITS = 50
        self.PAD_SIZE = 2
        
        
        # Train
        self.n_gpus = 1
        self.BATCH_SIZE = 32
        self.n_folds = 5
        self.n_epochs = 10
        self.model_file = './local/model.h5'
        self.model_bert_file = './keras_bert.h5'


        # Others
        self.stopwords_files = ['./data/京东商城商品评论-Stopwords.txt', 
                                './data/京东商城商品评论-Stopwords-other_github.txt']  # 公开停用词
        self.config_file = './local/config.pkl'     # config文件


    def set_pad_same_with_o(self, flag=True):
        TAGS = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
        self.TAGS = TAGS if flag else ['PAD'] + TAGS
        self.N_TAGS = len(self.TAGS)
        self.data_encoded_bert_file = './local/data_encoded_bert_' + str(self.N_TAGS) + '.pkl'   # 向量化编码后的训练数据 for BERT
        
        

def get_args():
    """待完善……"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--server',         default=None, type=int, help='[6099]')
    parser.add_argument('--phase',          default=None, help='[Train/Test]')
    parser.add_argument('--sen_len',        default=None, type=int, help='sentence length')

    parser.add_argument('--net_name',       default=None, help='[lstm]')
    parser.add_argument('--dir_date',       default=None, help='Name it with date, such as 20180102')
    parser.add_argument('--batch_size',     default=32, type=int, help='Batch size')
    parser.add_argument('--lr_base',        default=1e-3, type=float, help='Base learning rate')
    parser.add_argument('--lr_decay_rate',  default=0.1, type=float, help='Decay rate of lr')
    parser.add_argument('--epoch_lr_decay', default=1000, type=int, help='Every # epoch, lr decay lr_decay_rate')

    parser.add_argument('--layer_num',      default=2, type=int, help='Lstm layer number')
    parser.add_argument('--hidden_size',    default=64, type=int, help='Lstm hidden units')
    parser.add_argument('--n_gpus',         default='0', help='GPU id list')
    parser.add_argument('--workers',        default=4, type=int, help='Workers number')

    return parser.parse_args()



if __name__ == '__main__':
    
    args = get_args()
    n_gpus = args.n_gpus
    