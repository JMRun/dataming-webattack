##  类型      label
##  白         0
##  SQL注入    1
##  
##  远程代码   3
##  命令执行   4
##  XSS跨站脚本 5

##代码参考： http://t.csdnimg.cn/Kd8SF

##数据集生成
# -*- coding: utf-8 -*-
# pre.py

import numpy as np
import pandas as pd
from config import Config
from paddlenlp.datasets import load_dataset
import json
import jieba.analyse
import random
import os

class Pre(object):
    labels = {}
    labels_ = {}
    def __init__(self):
        self.cf = Config()
        self.dataPath = self.cf.dataPath
        self.trainRatio = self.cf.trainRatio

    def train(self):
        lists = []
        dataFolder = self.dataPath+'/train/'
        for filename in os.listdir(dataFolder):
            file_path =  os.path.join(dataFolder,filename)
            print(file_path)
            df = pd.read_csv(file_path).astype(str)
            print(len(df))
            #遍历每行
            for j in range(len(df)):
                item = []
                #读取每行的method列
                #从当前行中提取 method 字段，并添加到 item 列表
                item.append(df.loc[j, 'method'])
                #使用 jieba 提取 user_agent 列的前 10 个关键词，并将其添加到 item 列表中
                r0 = jieba.analyse.extract_tags(df.loc[j, 'user_agent'], topK=10)
                item.extend(r0)
                r1 = jieba.analyse.extract_tags(df.loc[j, 'url'], topK=20)
                item.extend(r1)
                r2 = jieba.analyse.extract_tags(df.loc[j, 'refer'], topK=10)
                item.extend(r2)
                r3 = jieba.analyse.extract_tags(df.loc[j, 'body'], topK=20)
                item.extend(r3)
                #创建一个字典 item_，包含合并后的文本（通过空格连接 item 列表的元素）和标签（lable 列的值）。
                item_ = {
                    'text': " ".join(item),
                    'label': df.loc[j, 'lable']
                }
                print(j,item_)
                lists.append(item_)
        #随机打乱 lists 列表中的数据顺序，防止数据的顺序影响训练效果
        random.shuffle(lists)
        print(len(lists))
        #计算训练集的大小，self.trainRatio 是训练集所占比例。
        offset = int(len(lists)*float(self.trainRatio))
        #划分训练集和验证集
        trains = lists[0:offset]
        valids = lists[offset:]

        # 生成新的数据集
        #打开（或创建）两个 JSON 文件，分别用于存储训练集和验证集
        trainsF = open(self.cf.minePath+'/train.json', 'w')
        validsF = open(self.cf.minePath+'/valid.json', 'w')
        #写入
        for item in trains:
            trainsF.write(json.dumps(item, ensure_ascii=False))
            trainsF.write('\n')

        for item in valids:
            validsF.write(json.dumps(item, ensure_ascii=False))
            validsF.write('\n')

    def test(self):
        tests = []
        df = pd.read_csv(self.dataPath+'/test/test.csv').astype(str)
        print(len(df))
        for j in range(len(df)):
            item = []
            item.append(df.loc[j, 'method'])
            r0 = jieba.analyse.extract_tags(
                df.loc[j, 'user_agent'], topK=10)
            item.extend(r0)
            r1 = jieba.analyse.extract_tags(df.loc[j, 'url'], topK=20)
            item.extend(r1)
            r2 = jieba.analyse.extract_tags(df.loc[j, 'refer'], topK=10)
            item.extend(r2)
            r3 = jieba.analyse.extract_tags(df.loc[j, 'body'], topK=20)
            item.extend(r3)
            item_ = {
                'id': df.loc[j, 'id'],
                'text': " ".join(item)
            }
            print(j, item_)
            tests.append(item_)
        print(len(tests))

        # 生成新的数据集
        testsF = open(self.cf.minePath+'/test.json', 'w')

        for item in tests:
            testsF.write(json.dumps(item, ensure_ascii=False))
            testsF.write('\n')
pre = Pre()
pre.train()
pre.test()


