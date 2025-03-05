# -*- coding:utf-8 -*-
from pyhanlp import *

import hanlp

def _pos(
        _doc_array,
        _method:str="crf",
):
    nature_list = []
    if _method in ("perceptron", "crf"):
        tokenizer = HanLP.newSegment(_method)
        for doc in _doc_array:
            nature_list.append([tokenizer.tag(doc)])
        return nature_list
    else:
        raise ValueError("错误的方法类型")

def Do_pos(doc_array,pos_method):
    nature_list = _pos(doc_array, pos_method)
   
def pos2vec(nature_list):
    i = 1
    table = dict()
    nature_vec = []
    for doc in nature_list:
        temp = []
        for nature in doc:
            if nature not in table:
                table[nature] = i
                i -=- 1
            temp.append(table[nature])
        nature_vec.append(temp)
    
    # print(table)
    # print(nature_vec)
    return nature_vec

if __name__ == "__main__":
    pass