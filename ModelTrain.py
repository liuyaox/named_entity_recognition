# -*- coding: utf-8 -*-
"""
Created:    2019-11-13 18:37:52
Author:     liuyao8
Descritipn: 
"""

from numpy import expand_dims
from collections import Counter
import pickle

from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model

from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_bert import Tokenizer

from model.NerBiLSTM import NerBiLSTM
from model.NerBiLSTMsTD import NerBiLSTMsTD
from model.NerBiLSTM_Bert import NerBiLSTM_Bert
from model.NerBiLSTMConvTD import NerBiLSTMConvTD
from Config import Config
config = Config()



# 数据预处理
def data_parsing(data_file):
    """解析语料文件"""
    samples = open(data_file, 'r', encoding='utf8').read().strip().split('\n\n')
    data = [[row.split() for row in sample.split('\n')] for sample in samples]
    return data


def data_encoding(data, word2idx, tags, maxlen):
    """数据编码  Y直接使用indices而非one-hot"""
    X = [[word2idx.get(str(w[0]).lower(), 1) for w in sample] for sample in data]
    X = pad_sequences(X, maxlen)
    Y = [[tags.index(w[1]) + 1 for w in sample] for sample in data]     # indices从1开始，以防与mask_value=0冲突
    Y = pad_sequences(Y, maxlen)    # X与Y的mask_value应该一致，因为貌似源代码里处理Y时使用的是处理X的那个mask
    Y = expand_dims(Y, 2)           # (, maxlen) --> (, maxlen, 1)  与CRF输出的shape一致
    return X, Y


def data_config_processing():
    """准备好data和config"""
    train = data_parsing(config.train_file)
    test = data_parsing(config.test_file)
    config.MAXLEN = max(len(sample) for sample in train)   # 100
    
    word_counts = Counter(row[0].lower() for sample in train for row in sample)   # 基于train
    vocab = [w for w, f in word_counts.items() if f >= config.MIN_FREQ]
    config.VOCABSIZE = len(vocab)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    config.word2idx = word2idx
    
    x_train, y_train = data_encoding(train, word2idx, config.TAGS, config.MAXLEN)
    x_test, y_test = data_encoding(test, word2idx, config.TAGS, config.MAXLEN)
    
    pickle.dump((x_train, y_train, x_test, y_test), open(config.data_encoded_file, 'wb'))
    pickle.dump(config, open(config.config_file, 'wb'))



data_config_processing()
config = pickle.load(open(config.config_file, 'rb'))
x_train, y_train, x_test, y_test = pickle.load(open(config.data_encoded_file, 'rb'))



# Model0: Random Embedding -> BiLSTM -> CRF
nerbilstm = NerBiLSTM(config)
model = nerbilstm.model
plot_model(model, to_file=nerbilstm.name + '.png', show_shapes=True)

model.compile(Adam(1e-2), loss=crf_loss, metrics=[crf_accuracy])
history = model.fit(x_train, y_train, batch_size=16, epochs=2, validation_data=(x_test, y_test))

model.compile(Adam(1e-3), loss=crf_loss, metrics=[crf_accuracy])
history = model.fit(x_train, y_train, batch_size=16, epochs=3, validation_data=(x_test, y_test))

model.compile(Adam(1e-4), loss=crf_loss, metrics=[crf_accuracy])
history = model.fit(x_train, y_train, batch_size=16, epochs=3, validation_data=(x_test, y_test))

model.compile(Adam(1e-5), loss=crf_loss, metrics=[crf_accuracy])
history = model.fit(x_train, y_train, batch_size=16, epochs=5, validation_data=(x_test, y_test))



# Model1: Random Embedding -> BiLSTMs -> TD(Dense) -> CRF
nerbilstmstd = NerBiLSTMsTD(config)
model = nerbilstmstd.model
model.compile('rmsprop', loss=crf_loss, metrics=[crf_accuracy])
history = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test))



# Model2: Random Embedding -> Concatenate(BiLSTM, Conv1D -> TD(Dense)) -> TD(Dense) -> CRF
nerbilstmconvtd = NerBiLSTMConvTD(config)
model = nerbilstmconvtd.model
model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_accuracy])
history = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test))



# Model3: BERT -> BiLSTM -> CRF
nerbertbilstm = NerBiLSTM_Bert(config)
model = nerbertbilstm.model
model.compile(optimizer=Adam(1e-4), loss=crf_loss, metrics=[crf_accuracy])


#预处理输入X
def PreProcessInputData(text):
    tokenizer = Tokenizer(vocab)
    word_labels = []
    seq_types = []
    for sequence in text:
        code = tokenizer.encode(first=sequence, max_len=config.SEQ_MAXLEN)
        word_labels.append(code[0])
        seq_types.append(code[1])
    return word_labels, seq_types


#预处理输入Y
def PreProcessOutputData(text):
    tags = []
    for line in text:
        tag = [0]
        for item in line:
            tag.append(int(label[item.strip()]))
        tag.append(0)
        tags.append(tag)

    pad_tags = pad_sequences(tags, maxlen=config.SEQ_MAXLEN, padding="post", truncating="post")
    result_tags = expand_dims(pad_tags, 2)
    return result_tags



# 其他
from keras.callbacks import ModelCheckpoint, Callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

checkpointer = ModelCheckpoint(filepath="bilstm_1102_k205_tf130.w", verbose=0, save_best_only=True, save_weights_only=True)
losshistory = LossHistory()



# 保存和加载模型 # TODO 只能使用keras_contrib的这个API么？
from keras_contrib.utils import save_load_utils
model_path = 'xxx'
save_load_utils.save_all_weights(model, model_path)     # 保存模型
save_load_utils.load_all_weights(model, model_path)     # 加载模型