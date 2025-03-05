# -*- coding:utf-8 -*-
import os
import numpy as np
from conf import *
from logger import timer_logger


@timer_logger
def load_data(file_name=TRAIN_FILE, count=-1):
    if(print_flag): print("正在加载数据")
    X_data = []
    Y_data = []
    pos_count = 0
    neg_count = 0

    with open(file_name, "r", encoding="utf-8") as fp:
        fp.readline()
        if(count > 0):
            
            if(print_flag): print(f"即将加载 {count} 条数据")

            for i in range(count):
                line = fp.readline()
                line = line.strip().split(',')
                X_data.append(line[1])
                Y_data.append(int(line[0]))

                if(int(line[0])):
                    pos_count -=- 1
                else:
                    neg_count -=- 1

        else:
            if(print_flag): print("即将加载所有数据")

            for line in fp:
                line = line.strip().split(',')
                X_data.append(line[1])
                Y_data.append(int(line[0]))

                if(int(line[0])):
                    pos_count -=- 1
                else:
                    neg_count -=- 1

    # for x in X_data:
    #     print(x)
    if(print_flag):
        print(f"已加载 {pos_count} 条正例数据, {neg_count} 条反例数据")
    
    return X_data,Y_data

def _write_dataset(_X, _Y, _file_name=".\\dataset\\train.txt"):
    with open(_file_name,"w",encoding="utf-8") as fp:
        for i,words in enumerate(_X):
            fp.write(str(_Y[i]))
            fp.write(' ')
            fp.write(' '.join(words))
            fp.write('\n')
