# -*- coding:utf-8 -*-
import datetime
import sys

from load_data import *
from load_data import _write_dataset
from segment import *
from word2vec import *
from conf import *
from ml_models import ML_Models
from dl_models import LSTM_load_and_test
from part_of_speech import _pos,pos2vec


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def Enable_Logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,"w") as fp:
            pass
    sys.stdout = Logger(LOG_FILE, sys.stdout)
    pass

def _check_paths():
    if not os.path.exists(".\\logs"):
        os.mkdir(".\\logs")
    if not os.path.exists(".\\processed_data"):
        os.mkdir(".\\processed_data")

def load_and_test():
    # print("正在加载数据")
    # train_X, train_Y = load_data(TRAIN_FILE)
    train_X, train_Y = load_data(".\\data3\\train_balanced.txt")
    test_X, test_Y = load_data(".\\data3\\test_balanced.txt")

    # print("正在分词")
    '''
    method: crf/dat/viterbi/perceptron/cnn/bert/electra
    '''
    seg_method = "viterbi"
    rm_stop_words = True
    train_X = Do_Segment(train_X, seg_method, rm_stop_words, caller="train_X_balanced")
    test_X = Do_Segment(test_X, seg_method, rm_stop_words, caller="test_X_balanced")

    nature = False
    # print("正在词性标注")
    # train_X_nature = _pos(train_X,"crf")
    # test_X_nature = _pos(test_X,"crf")
    # train_X_nvec = pos2vec

    # _write_dataset(train_X, train_Y, ".\\dataset\\train_balanced.txt")
    # _write_dataset(test_X,test_Y,".\\dataset\\test_balanced.txt")

    # 使用LSTM模型不需要向量化
    # dl_model = LSTM_load_and_test(train_X,train_Y,test_X,test_Y)

    print("正在转换为向量")

    train_X_vec, test_X_vec = Do_vectorization(train_X, test_X, "tfidf")

    ml_model = ML_Models(train_X_vec,train_Y,test_X_vec,test_Y,)
    ml_model.use_model(model_name="rf")
    ml_model.train()

    ml_model.report()
    # print(sklearn_tfidf(train_X).shape)

if __name__ == "__main__":
    _check_paths()
    Enable_Logger()
    
    load_and_test()
    
    pass