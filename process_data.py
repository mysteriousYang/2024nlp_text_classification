# -*- coding:utf-8 -*-
import os
from conf import *

def balace_data(
        src_file=TEST_FILE, 
        dst_file=".\\data3\\test_balanced.txt"
):
    MAX_POS = 256
    MAX_NEG = 2000
    pos_count = 0
    neg_count = 0
    balaced = []

    with open(src_file, "r", encoding="utf-8") as fp:
        fp.readline()
        for line in fp:
            sample = line.strip().split(',')

            if(int(sample[0])):
                pos_count-=-1
                if(pos_count<MAX_POS):
                    balaced.append(line)

            else:
                neg_count-=-1
                if(neg_count<MAX_NEG):
                    balaced.append(line)

    with open(dst_file, "w", encoding="utf-8") as fout:
        for line in balaced:
            fout.write(line)

if __name__ == "__main__":
    balace_data()