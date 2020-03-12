# -*- coding: utf-8 -*-
"""
Created:    2019-11-13 18:37:52
Author:     liuyao8
Descritipn: 
"""
import pickle
from keras.utils import plot_model
from keras.optimizers import Adam
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

from utils import pre, rec, print_metrics

from model.NerBiLSTMBert import NerBiLSTMBert
from model.NerBiLSTM import NerBiLSTM
from model.NerBiLSTMsTD import NerBiLSTMsTD
from model.NerBiLSTMConvTD import NerBiLSTMConvTD



if __name__ == '__main__':

    from Config import Config
    config = Config()
    config = pickle.load(open(config.config_file, 'rb'))
    config.set_pad_same_with_o(False)
    
    
    # 1. Bert
    x_train, y_train, x_test, y_test = pickle.load(open(config.data_encoded_bert_file, 'rb'))
    
    nerbilstmbert = NerBiLSTMBert(config)

    nerbilstmbert.model.compile(optimizer=Adam(1e-2), loss=crf_loss, metrics=[crf_accuracy, pre, rec])
    history1 = nerbilstmbert.model.fit(x=x_train, y=y_train, batch_size=32, epochs=2, validation_split=0.3)

    nerbilstmbert.model.compile(optimizer=Adam(1e-3), loss=crf_loss, metrics=[crf_accuracy, pre, rec])
    history2 = nerbilstmbert.model.fit(x=x_train, y=y_train, batch_size=32, epochs=2, validation_split=0.3)

    nerbilstmbert.model.compile(optimizer=Adam(1e-4), loss=crf_loss, metrics=[crf_accuracy, pre, rec])
    history3 = nerbilstmbert.model.fit(x=x_train, y=y_train, batch_size=32, epochs=3, validation_split=0.3)

    nerbilstmbert.model.compile(optimizer=Adam(1e-5), loss=crf_loss, metrics=[crf_accuracy, pre, rec])
    history4 = nerbilstmbert.model.fit(x=x_train, y=y_train, batch_size=32, epochs=5, validation_split=0.3)
    
    nerbilstmbert.model.evaluate(x_test, y_test)
    y_pred = nerbilstmbert.model.predict(x_test)
    print_metrics(y_test, y_pred)


    sentence = 'xxxxxx'
    tags = nerbilstmbert.model_predict(sentence, dp)

    
    
    # 2. 非Bert
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