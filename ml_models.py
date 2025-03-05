# -*- coding:utf-8 -*-
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from typing import List

import torch

from logger import timer_logger
from conf import print_flag


class ML_Models:
    '''
    构造函数: 必须传入已经矢量化后的训练集与测试集
    '''
    def __init__(
            self,
            train_X,
            train_Y,
            test_X,
            test_Y,

        ):
            self.train_X = train_X
            self.train_Y = train_Y
            self.test_X = test_X
            self.test_Y = test_Y
            self.model = None
            pass
    
    def use_model(self, **kwargs):
        '''
        参数可变, 请传入model_name使用特定模型
        目前支持的模型有
        基于sklearn: svc/mlp/lr/bayes/rf
        基于hmmlearn: hmm(暂未开发)

        其他可能的参数:
        核函数 kernel:str
        隐层结构 hidden_layer_size:tuple
        激活函数 activation:str
        最优值解算 solver:str
        学习率 alpha:float
        最大树深 max_depth:int
        最小分割样本量 min_samples_split:int
        '''

        if("model_name" in kwargs):
            if(kwargs["model_name"] == "svc"):
                '''
                主要使用参数:
                核函数 kernel:str
                '''
                kernel = "rbf"
                if("kernel" in kwargs): kernel = kwargs["kenerl"]
                self.model = SVC(kernel=kernel)

            elif(kwargs["model_name"] == "mlp"):
                '''
                主要使用参数:
                隐层结构 hidden_layer_size:tuple
                激活函数 activation:str
                最优值解算 solver:str
                学习率 alpha:float
                '''
                hidden_layer = (50,30,15,2)
                activation = "relu"
                solver = "adam"
                alpha = 1e-2

                if("hidden_layer_size" in kwargs):
                    hidden_layer = kwargs["hidden_layer_size"]
                if("activation" in kwargs):
                    activation = kwargs["activation"]
                if("solver" in kwargs):
                    solver = kwargs["solver"]
                if("alpha" in kwargs):
                    alpha = kwargs["alpha"]

                self.model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer,
                    activation=activation,
                    solver=solver,
                    alpha=alpha,
                )

            elif(kwargs["model_name"] == "lr"):
                '''
                
                '''
                self.model = LogisticRegression()

            elif(kwargs["model_name"] == "rf"):
                '''
                最大树深 max_depth:int
                最小分割样本量 min_samples_split:int
                '''
                max_depth = None
                min_samples_split = 2
                if("max_depth" in kwargs): max_depth = kwargs["max_depth"]
                if("min_samples_split" in kwargs): min_samples_split = kwargs["min_samples_split"]

                self.model = ExtraTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split)
            
            elif(kwargs["model_name"] == "bayes"):
                '''
                
                '''
                self.train_X = self.train_X.toarray()
                self.test_X = self.test_X.toarray()
                self.model = GaussianNB()
            pass
        else:
            #默认使用SVC
            self.model = SVC()
            pass


    @timer_logger
    def train(
        self,
        train_X:torch.Tensor=None,
        train_Y:torch.Tensor=None,
    ):
        '''
        如果不传入任何参数即使用初始化的数据训练
        '''
        if(print_flag): print("正在训练模型")

        if(train_X and train_Y):
            self.model.fit(train_X, train_Y)
        else:
            self.model.fit(self.train_X, self.train_Y)

    def report(self):
        pred_Y = self.model.predict(self.test_X)

        print( f"使用模型 {self.model} 的分类结果:\n" ,
                classification_report(self.test_Y, pred_Y)
        )
        
        print("混淆矩阵\n",
              confusion_matrix(self.test_Y, pred_Y)
        )
        pass