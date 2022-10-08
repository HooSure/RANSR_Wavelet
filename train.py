from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import random

from utils.utils import *
import models.ModelRAN as ModelRAN
import argparse
import time
import torch

import utils.utils as zxhUtils
import utils.image_ronghe as zxhImgUnited
import sys
#import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def reOpts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_targetData',type=str, default='dataSet/targetData/targetData_level-2_0.xlsx', help='目标样本地址')
    parser.add_argument('--src_noiseData', type=str, default='dataSet/inputData/inputData_level-2_0.xlsx', help='带改造样本地址')
    parser.add_argument('--src_testData', type=str, default='dataSet/testData/test.xlsx', help='测试样本地址')
    parser.add_argument('--data_Number', type=int, default=1563, help='数据表总量')
    parser.add_argument('--src_RAN_model',type=str, default='RANs/RAN_model', help='模型权重保存地址')
    parser.add_argument('--src_RAN_data', type=str, default='RANs/RAN_data', help='模型样本保存地址')
    parser.add_argument('--src_RAN_test', type=str, default='RANs/RAN_test', help='测试生成样本地址')

    parser.add_argument('--data_rows', type=int, default=1, help='数据行数')
    parser.add_argument('--data_cols', type=int, default=123, help='数据列数')
    parser.add_argument('--channels', type=int, default=1, help='数据通道数')
    parser.add_argument('--latent_dim', type=int, default=20, help='生成器输入大小')
    parser.add_argument('--sample_size', type=int, default=1280, help='样本量大小')
    parser.add_argument('--sampleSize', type=int, default=256, help='一张图像样本量大小')
    parser.add_argument('--count', type=int, default=1, help='量化大小')
    parser.add_argument('--count_noise', type=int, default=1, help='带改造样本量化大小')
    
    parser.add_argument('--epochs', type=int, default=10001, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=128, help='每轮训练数据量(类似图片量)')
    parser.add_argument('--sample_lenth', type=int, default=20, help='单个训练数据')
    parser.add_argument('--sample_interval', type=int, default=500, help='输出数据间隔')
    opt = parser.parse_args()
    return opt

def showTime(func):
    import time
    #装饰器函数：显示运行时间
    def wrapper(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        #print(args[0])
        endTime = time.time()
        msecs = (endTime - startTime)*1000
        print("time is %d ms" %msecs)
    return wrapper

if __name__ == '__main__':
    #zxhImgUnited.runTest('RANs/RAN_data/2022-10-06 21-1/origin.jpg','RANs/RAN_data/2022-10-06 21-1/outputName3.jpg',1,1)
    #sys.exit()
    opts = reOpts()
    ran = ModelRAN.RAN(opts)

    #print(ran)
    #ran.train(20001)
    #sys.exit()
    zxhUtils.overSizeImageRes('img_021_SRF_2_HR.jpg','test.jpg',2)
    ran.forecast('test_overSizelevel2',2599,-1,'outputName001')
    sys.exit()
    for cunt in range(0,2):
        ran.foreCast('outputName', 4999, -2, 'outputName{}'.format(cunt))




















    '''time.sleep(1)
    ModelRAN.test()
    sys.exit()'''
    '''#ran = RAN(reOpts())
    #ran.train(epochs=ran.opt.epochs)
    #ran.test()
    dataSet = ModelRAN.GetDataset(0)
    print(dataSet.type)'''
    #checkPath("bbb",1)
    #ModelRAN.test()
    dataSet = ModelRAN.GetDataset(0,123,20,128)
    target, input = dataSet.getNxetDataset()
    test ,testsum = dataSet.getTestData()
    print('target.shape',target.shape,'input.shape',input.shape,'input.max',torch.max(input).item())#torch.max(dataSet.Normalize(input)).item()
    print("targetDataMax：",dataSet.targetDataMax)
    print("inputDataMax：", dataSet.inputDataMax)
    for _ in range(100):
        target, input = dataSet.getNxetDataset()
        print("targetDataMax：", dataSet.targetDataMax)
        print("inputDataMax：", dataSet.inputDataMax)
        print("差的倍数：", dataSet.targetDataMax/dataSet.inputDataMax)

    sys.exit()
    print('target.type', target.type, 'input.type', input.type)
    print('test[0].shape', test[0].shape, 'test[1].type', test[0].type)
    print('testsum.shape', testsum.shape, 'testsum.type', testsum.type)
    '''ran = RAN()
    ran.train(epochs=20001, batch_size=16, sample_interval=500)
    ran.test()'''

