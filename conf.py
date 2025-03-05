# -*- coding:utf-8 -*-
import datetime

print_flag = True

TRAIN_FILE = ".\\data3\\train.txt"
TEST_FILE = ".\\data3\\test.txt"
DEV_FILE = ".\\data3\\dev.txt"

PROC_DIR = ".\\processed_data\\"

LOG_FILE = ".\\logs\\" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".log"

BERT_MODEL_PATH = ".\\models\\sighan2005_pku_bert_base_zh_20201231_141130"