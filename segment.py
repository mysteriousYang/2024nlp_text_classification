# -*- coding:utf-8 -*-
from functools import wraps
from pyhanlp import *
from enum import Enum
from typing import Optional,List
import os
import hanlp

from load_data import load_data
from conf import *
from logger import timer_logger

class Segment_Method(Enum):
    "dat"
    "crf"
    "viterbi"
    "perceptron"
    "basic"
    "nlp"
    "cnn"
    "bert"
    "electra"

def _seg_file_name(_caller_:str, _method:str, _stwd_flag:bool):
    if(_stwd_flag):
        return f"{_caller_}-{_method}-removed.txt"
    return f"{_caller_}-{_method}.txt"

def _save_segment(_caller:str, _method:str, _stwd_flag:bool, _termlist:List[List[str]]):
    with open(PROC_DIR+_seg_file_name(_caller,_method,_stwd_flag), 'w', encoding="utf-8") as fp:
        for array in _termlist:
            fp.write(' '.join(array))
            fp.write('\n')


def _load_segment(_caller:str, _method:str, _stwd_flag:bool):
    termlist = []
    with open(PROC_DIR+_seg_file_name(_caller, _method,_stwd_flag), 'r', encoding="utf-8") as fp:
        for line in fp:
            termlist.append(line.strip().split(' '))
    return termlist

@timer_logger
def segment(
        _text_array:list, 
        _method:Segment_Method):
    X_data = []

    if(print_flag): print(f"正在使用 {_method} 方法分词")

    if _method in ("crf","perceptron"):
        tokenizer = HanLP.newSegment(_method)
        for text in _text_array:
            # .seg接口     分词并标注词性
            # X_data.append(tokenizer.seg(text))
            # .segment接口 只分词
            X_data.append(list(tokenizer.segment(text)))
        pass

    elif _method in ("dat","viterbi"):
        tokenizer = HanLP.newSegment(_method)
        termlist = []
        for text in _text_array:
            #这两个类没有仅分词的接口
            termlist.append(list(tokenizer.seg(text)))
        X_data = [[term.word for term in sentence] for sentence in termlist]

    elif _method in ("basic","nlp"):
        termlist = []
        if _method == "basic":
            tokenizer = JClass("com.hankcs.hanlp.tokenizer.BasicTokenizer")
            #X element: com.hankcs.hanlp.seg.common.Term
        else:
            tokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
            #X element: com.hankcs.hanlp.seg.common.Term
        for text in _text_array:
            termlist.append(list(tokenizer.segment(text)))
        #这两个也没有仅分词的接口
        X_data = [[term.word for term in sentence] for sentence in termlist]
            
        # print(X_data[0][0].nature)
        pass

    elif _method in ("bert","cnn","electra"):
        # print("该接口尚未开发(2024/6/18)")
        # exit(2)
        if _method == "cnn":
            tok = hanlp.load(hanlp.pretrained.tok.PKU_NAME_MERGED_SIX_MONTHS_CONVSEG)#一个CONV模型
            
        elif _method == "bert":
            # tok = hanlp.load(hanlp.pretrained.tok.SIGHAN2005_PKU_BERT_BASE_ZH)
            os.environ["http_proxy"] = "http://127.0.0.1:7890"
            os.environ["https_proxy"] = "http://127.0.0.1:7890"
            tok = hanlp.load(BERT_MODEL_PATH)

        elif _method == "electra":
            tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)


        X_data = tok(_text_array)
        # print(X_data)
        return X_data
        pass
    else:
        print("分词方法错误")
        exit(1)

    # print(X_data)
    return X_data
    
@timer_logger
def remove_stop_words(
        text_array,
        stop_words_dir="stopwords.txt"
):
    # 适用于termlist的版本
    # CoreStopWordDictionary = JClass("com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary")
    # for sentence in termlist:
    #     CoreStopWordDictionary.apply(sentence)
    stop_words = set()
    meaningful_words = list()
    with open(stop_words_dir, "r", encoding="utf-8") as fp:
        stop_words = set(word.strip() for word in fp.readlines())
    if(print_flag): print("正在处理停用词")
    for sentence in text_array:
        meaningful_words.append([word for word in sentence if len(word) > 1 and word not in stop_words])
    return meaningful_words


def Do_Segment(
        text_array:list, 
        segment_method:Segment_Method="viterbi", 
        rm_stop_words:bool=False,
        stop_words_dir="stopwords.txt",
        caller=""
):
    path = PROC_DIR + _seg_file_name(caller, segment_method, rm_stop_words)

    #如果可以读取,则直接读取
    if(os.path.exists(path)):
        if(print_flag):
            print("正在从文件加载分词结果")
            print(f"读取文件 {path}...")

        return _load_segment(caller, segment_method, rm_stop_words) 

    #分词后自动保存
    else:
        termlist = segment(text_array, segment_method)
        if(rm_stop_words):
            termlist =  remove_stop_words(termlist)
        
        _save_segment(caller, segment_method, rm_stop_words, termlist)
        return termlist

def Term_To_List(termlist):
    # if(with_nature):
    #     L = []
    #     for sentence in termlist:
    #         L_word = []
    #         L_pos = []
    #         for term in sentence:
    #             L_word.append(term.word)
    #             L_pos.append(str(term.nature))
    #         L.append(L_word + L_pos)
    #     return L

    return [[term.word for term in sentence] for sentence in termlist]


if __name__ == "__main__":

    # X_data, Y_data = load_data()
    # X_data = Do_Segment(X_data, "adt")
    # with open("segment_out.txt","w",encoding="utf-8") as fout:
    #     for array in X_data:
    #         fout.write(str(array))
    #         fout.write('\n')
    # # print(X_data)
    pass