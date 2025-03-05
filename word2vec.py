# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from pyhanlp import *
from typing import List,Tuple
import hanlp
import gensim
import torch

from conf import *
from segment import *
from load_data import *
from logger import timer_logger

def Do_vectorization(
        train_X:List[List[str]],
        test_X:List[List[str]],
        method:str
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    文本矢量化函数
    doc_array: 二维单词列表
    method: doc2vec / count / tfidf
    '''

    #使用hanlp的transformer
    if(method == "doc2vec"):
        tr_vec = hanlp_doc2vec(train_X)
        ts_vec = hanlp_doc2vec(test_X)
        return tr_vec, ts_vec
    
    #如果使用sklearn
    #这一步要把train和test加起来一起转换
    elif(method == "count"):
        all_vec = sklearn_count_vec(train_X + test_X)
        tr_vec = all_vec[:len(train_X)]
        ts_vec = all_vec[len(train_X):]
        return tr_vec, ts_vec
    
    elif(method == "tfidf"):
        all_vec = sklearn_tfidf(train_X + test_X)
        tr_vec = all_vec[:len(train_X)]
        ts_vec = all_vec[len(train_X):]
        return tr_vec, ts_vec
    else:
        raise ValueError("错误的矢量化方法")



@timer_logger
def hanlp_doc2vec(doc_array:List[List[str]]):
    if(print_flag): print("正在使用hanlp的doc2vec接口")

    word2vec = hanlp.load(hanlp.pretrained.word2vec.CONVSEG_W2V_NEWS_TENSITE_WORD_PKU)
    # 这里应该可以不用doc2vec参数
    # 它的操作只是把整个张量求均值
    # 这里使用 tensor.mean(dim=0) 可以获得每个特征维度的均值

    vec_array = torch.empty( size=(len(doc_array), 50) )
    for i,doc in enumerate(doc_array):
        vec = word2vec(doc).mean(dim=0)
        #非常值处理
        vec = torch.where(torch.isnan(vec), torch.full_like(vec, 0), vec)
        vec = torch.where(torch.isinf(vec), torch.full_like(vec, 1), vec)

        vec_array[i] = vec

    # print(vec_array.shape) 

    return vec_array
    pass


@timer_logger
def sklearn_count_vec(
        doc_array:List[List[str]]=[],
):
    if(print_flag): print("正在使用sklearn的词袋接口")
    #sklearn接口不同,需要提前对数据进行处理
    #虽然其实sklearn自己直接就可以完成分词去除重复词的任务
    split_doc_list = []

    for array in doc_array:
        split_doc_list.append(" ".join(array))

    # print(split_doc_list)

    vectorizer = CountVectorizer()
    result = vectorizer.fit_transform(split_doc_list)

    # print(result.toarray())

    return result
    

@timer_logger
def sklearn_tfidf(
        doc_array:List[List[str]]=[],
):
    if(print_flag): print("正在使用sklearn的tfidf接口")

    split_doc_list = []

    for array in doc_array:
        split_doc_list.append(" ".join(array))

    vectorizer = TfidfVectorizer()
    result = vectorizer.fit_transform(split_doc_list)

    # print(result.toarray())

    return result


def _build_gensim_dataset(
        doc_array=[]
):
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    doc_dataset = list()
    for idx, doc in enumerate(doc_array):
        doc_dataset.append(TaggededDocument(doc,[idx]))

    
    return doc_dataset

def _train_gensim_doc2vec(
        doc_dataset=[],
        save_dir=".\\models\\gs_doc2vec.model",
):
    model = gensim.models.doc2vec.Doc2Vec(doc_dataset,min_count=1,window=5,workers=16)
    model.train(doc_dataset,total_examples=len(doc_dataset),epochs=10)
    model.save(save_dir)


if __name__ == "__main__":
    # count_vec()
    train_X,train_Y = load_data(TRAIN_FILE,10)
    seg_method = "crf"
    rm_stop_words = True
    train_X = Do_Segment(train_X, seg_method, rm_stop_words)

    # hanlp_tryDoc2vec()
    hanlp_doc2vec(train_X)
    # print("Building Dataset")
    # gensim_dataset = _build_gensim_dataset(train_X)
    # print("Start training")
    # _train_gensim_doc2vec(gensim_dataset)

    pass