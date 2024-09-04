# -*- coding: utf-8 -*-
class Config(object):
    def __init__(self):
        #/home/jiliqiang/DataMining/webAttackDetection/webAt/data
        self.dataPath = './data' # 原始数据集路径
        self.minePath = './webat1/generateData' # 生成的数据集路径
        self.trainRatio = '0.8' # 训练集和验证集比例
        self.modelPath = './webat1/model/'
        self.rankPath = './webat1/rank/'
        self.logPath = './webat1/logs' 
        self.pointsPath = './webat1/checkpoint' # 每轮保存的权重
        self.inferencePath = './webat1/inference/'
        self.pointsPath = './webat1/checkpoint' # 每轮保存的权重
        self.inferencePath = './webat1/inference/'
        self.use_gpu = 1 # 是否使用GPU
