# -*- coding:utf-8 -*-
import time


def timer_logger(func):
    def wrapper(*args, **kwargs):
        start = time.clock()
        result = func(*args,**kwargs)
        print(f"运行时间: {time.clock() - start}s")
        return result
    return wrapper
