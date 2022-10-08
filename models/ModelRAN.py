from __future__ import print_function, division
'''from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam'''
import pandas as pd
#import matplotlib.pyplot as plt
#import sys
import numpy as np
##import xlrd
import random
#from utils.utils import *
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import sys
import os

import torch.autograd
from torch.autograd import Variable
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import cv2


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def test():
    sys.path.append(os.path.dirname(__file__) + os.sep + '../')
    from utils.utils import checkPath
    checkPath('uuu')

def resTime():
    import time
    startTime = time.time()
    timeArray = time.localtime(startTime)  # timeStamp_13 / 1000
    otherStyleTime = time.strftime("%Y-%m-%d %H-%M-%S", timeArray)
    return otherStyleTime

def showTime(func):
    import time
    #装饰器函数：显示运行时间
    def wrapper(*args, **kwargs):
        startTime = time.time()
        timeArray = time.localtime(startTime)  # timeStamp_13 / 1000
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        print("startTime is %s " % otherStyleTime)
        func(*args, **kwargs)
        #print(args[0])
        endTime = time.time()
        msecs = (endTime - startTime) * 1000
        print("time is %d ms" % msecs)
    return wrapper

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


class GetDataset(Dataset):

    def __init__(self,pathNumber,data_cols=123,latent_dim=20,batch_size = 128):
        # Initialize data, download, etc.
        # read with numpy or pandas
        self.data_cols = data_cols
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.inputDataMax = 1.0
        self.targetDataMax = 1.0
        self.testDataMax = 1.0

        self.renewData(pathNumber)#获取data和

        self.n_samples = self.data.shape[0]

        # here the first column is the class label, the rest are the features
        #print('input_data.shape:', self.noise_data.shape)
        #print('targetData.shape:',self.data.shape)
        print('GetDataset init！')
        #self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        #self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

    def getSempleInterval(self):
        idy = random.randint(0,self.data_cols-self.latent_dim)
        return idy


    def getNxetDataset(self):
        idx = np.random.randint(0, self.data.shape[0], self.batch_size)
        idy = self.getSempleInterval()
        # 从数据集随机挑选batch_size个数据，作为一个批次训练
        target = self.data[idx, :, idy:idy + self.latent_dim]
        input = self.noise_data[idx, :, idy:idy + self.latent_dim]

        return self.Normalize(torch.from_numpy(target)),self.Normalize(torch.from_numpy(input),0)

    def renewData(self,sumNumber = 1563):
        # 更新数据
        # read with numpy or pandas
        pathNumber = random.randint(0, sumNumber)
        targetData = pd.read_excel('./dataSet/targetData/targetData_level-2_%d.xlsx'%pathNumber, header=None)
        inputData = pd.read_excel('./dataSet/inputData/inputData_level-2_%d.xlsx' % pathNumber, header=None)
        self.data = np.array(targetData.values.tolist()).reshape(-1, 1,self.data_cols)  # (self.sample_size, 1, self.latent_dim)
        self.noise_data = np.array(inputData.values.tolist()).reshape(-1, 1,self.data_cols)  # (self.sample_size, 1, self.latent_dim)
        
    def getTestData(self,fromExcel = True,TestData = 'none'):
        testNxetData = []
        if fromExcel:
            TestData = pd.read_excel('./dataSet/testData/test.xlsx', header=None)
            testdata = np.array(TestData.values.tolist()).reshape(-1, 1, self.data_cols)
            for i in range(int(self.data_cols / self.latent_dim)):
                idy = i * self.latent_dim
                testNxetData.append(
                    self.Normalize(torch.from_numpy(testdata[:, :, idy:idy + self.latent_dim]), 2))  # 256*1*123
        else:
            testdata = np.array(TestData.tolist()).reshape(-1, 1,TestData.shape[-1])
            #idy = 0
            for i in range(int(testdata.shape[-1]/self.latent_dim)):
                idy = i * self.latent_dim
                testNxetData.append(self.Normalize(torch.from_numpy(testdata[:, :, idy:idy + self.latent_dim]),2))#256*1*123
                #idy += self.latent_dim
        return testNxetData, torch.from_numpy(testdata)


    def Normalize(self,inData,dataIsTarget = 1):
        '''
        数据归一化至[-1,1]并返回
        0:保存带改造权值
        1:保存目标权值
        2:测试权值
        其他：
        '''
        maxTemp = max(torch.max(inData).item(),abs(torch.min(inData).item()))
        if dataIsTarget == 1:
            self.targetDataMax = maxTemp
        if dataIsTarget == 0:
            self.inputDataMax = maxTemp
        if dataIsTarget == 2:
            self.testDataMax = maxTemp
        return (inData / maxTemp) * 0.99

    def NormalizeToCompare(self,inData,targetMultiple = 1):
        '''
        [-1,1]转化到[0,1],这个范围才能使用BCELoss的损失函数to Compare
        '''
        return (inData + targetMultiple) / (targetMultiple * 2)
    
    def NormalizeToMax(self,inData,dataIsTarget = 1,targetMultiple = 1):
        '''
        归一化数据复原
        从[0 - 1]到[-maxTemp - +maxTemp]
        '''
        maxTemp = 1
        if dataIsTarget == 1:
            maxTemp =self.targetDataMax
        if dataIsTarget == 0:
            maxTemp= self.inputDataMax
        if dataIsTarget == 2:
            maxTemp = self.testDataMax
        return ((inData * 2) - targetMultiple) * maxTemp

# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
class discriminator(nn.Module):
    def __init__(self,latent_dim=123):
        super(discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.dis = nn.Sequential(
            nn.Linear(self.latent_dim, 256),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Conv1d(1, 1,3, padding=1),  # 进行一个卷积操作
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2, stride=2),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
            # sigmoid可以把实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )

    def forward(self, x):
        x = self.dis(x)
        return x

# ###### 定义改造器 Reinventor #####
# 输入一个123维的0～1之间的图像，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
class reinventor(nn.Module):
    def __init__(self,latent_dim=123):
        super(reinventor, self).__init__()
        self.latent_dim = latent_dim
        self.rein = nn.Sequential(
            nn.Conv1d(1, 1,kernel_size=3, padding=1),  # 进行一个卷积操作
            nn.LeakyReLU(0.5),  # relu激活nn.Tanh()  ,#nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim, 256),  # 用线性变换将输入映射到256维
            nn.LeakyReLU(0.5),   # relu激活nn.Tanh(),  # nn.LeakyReLU(0.2),#nn.ReLU(True),  # relu激活
            nn.Linear(256, 1024),  # 用线性变换将输入映射到256维
            nn.LeakyReLU(0.5),  # relu激活nn.Tanh(),
            nn.MaxPool1d(2, stride=2),
            nn.LeakyReLU(0.5), # relu激活nn.Tanh(),  # nn.LeakyReLU(0.2),#
            nn.Linear(512, 256),  # 用线性变换将输入映射到256维
            nn.LeakyReLU(0.5),  # relu激活nn.Tanh(),  # nn.LeakyReLU(0.2),#nn.ReLU(True),  # relu激活

            nn.Linear(256, 2*self.latent_dim),  # 线性变换
            nn.MaxPool1d(2, stride=2),
            nn.Tanh(),# relu激活nn.Sigmoid()  # Tanh激活使得生成数据分布在【-1,1】之间，因为输入的真实数据的经过transforms之后也是这个分布
        )

    def forward(self, x):
        x = self.rein(x)
        return x

class RAN():
    def __init__(self, opt):
        self.opt = opt
        self.data_rows = opt.data_rows
        self.data_cols = opt.data_cols
        self.channels = opt.channels
        self.data_shape = (self.data_rows, self.data_cols)

        self.latent_dim = opt.latent_dim
        self.sample_shape = (self.data_rows, self.latent_dim)
        self.sample_size = opt.sample_size
        self.sampleSize = opt.sampleSize
        self.batch_size = opt.batch_size

        self.count = opt.count
        self.count_noise = opt.count_noise

        self.src_targetData = opt.src_targetData
        self.src_noiseData = opt.src_noiseData
        self.src_testData = opt.src_testData
        self.data_Number = opt.data_Number
        self.src_RAN_model = opt.src_RAN_model
        self.src_RAN_data = opt.src_RAN_data
        self.src_RAN_test = opt.src_RAN_test
        self.testNumber = 5500  # 测试时用的权重




    @showTime
    def train(self, epochs = 10001, batch_size=128, sample_interval=500):
        #数据加载
        dataSet = GetDataset(0, self.data_cols, self.latent_dim, self.batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        writer = SummaryWriter('./runs/RANs')
        # 创建对象
        #D = discriminator(self.latent_dim)
        R = reinventor(self.latent_dim)
        #D = D.to(device)
        R = R.to(device)
        '''if torch.cuda.is_available():
            D = D.cuda()
            R = R.cuda()'''

        # 首先需要定义loss的度量方式  （二分类的交叉熵）
        # 其次定义 优化函数,优化函数的学习率为0.0003
        criterion = nn.MSELoss(reduce=True, size_average=True)#BCELoss()  # 损失函数
        #criterion = nn.BCEWithLogitsLoss()
        #criterion = nn.SmoothL1Loss()
        #d_optimizer = torch.optim.SGD(D.parameters(), lr=0.002, momentum=0.9)#Adam(D.parameters(), lr=0.0005)#
        #R_optimizer = torch.optim.Adam(R.parameters(), lr=0.002)#SGD(R.parameters(), lr=0.0005, momentum=0.9)
        R_optimizer = torch.optim.Adam(R.parameters(), lr=0.002)#Adam(R.parameters(),0.001)SGD(R.parameters(), lr=0.002, momentum=0.9)#
        #real_label = Variable(torch.ones(batch_size)).to(device) # 定义真实的图片label为1
        #fake_label = Variable(torch.zeros(batch_size)).to(device)  # 定义假的图片的label为0

        #if torch.cuda.is_available():

        # ##########################进入训练##判别器的判断过程#####################
        for epoch in range(epochs):  # 进行多个epoch的训练
            for i in range(1000):

                # view()函数作用是将一个多行的Tensor,拼接成一行
                # 第一个参数是要拼接的tensor,第二个参数是-1
                # =============================训练判别器==================
                #num_img = img.size(0)
                # #img = img.view(num_img, -1)  # 将图片展开为28*28=784
                if i %50 == 0:
                    target, input_ = dataSet.getNxetDataset()
                    target = dataSet.Normalize(target,1)
                    real_img = Variable(target.type(torch.float32)).to(device)  # 将tensor变成Variable放入计算图中
                '''if torch.cuda.is_available():
                    real_img = Variable(input).cuda()  # 将tensor变成Variable放入计算图中'''
                #print(real_img.shape)
                z = Variable(input_.type(torch.float32)).to(device)  # 随机生成一些噪声
                fake_img = R(z)  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
                #fake_img = dataSet.Normalize(fake_img, 0)
                #print('mean', fake_img.mean().item(),'min', fake_img.min().item(),'max', fake_img.max().item())
                fake_img = dataSet.Normalize(fake_img,3)
                #print('mean', fake_img.mean().item(),'min', fake_img.min().item(),'max', fake_img.max().item())
                '''if not (fake_img.max().item() > -1):
                    print('v', fake_img.values().item())
                if i % 200 == 0:
                    print('v', fake_img.detach().cpu().numpy())'''
                    #print('min',fake_img.min().item())
                    #print('max', fake_img.max().item())
                #continue
                #sys.exit()
                R_loss = criterion(dataSet.NormalizeToCompare(fake_img), dataSet.NormalizeToCompare(real_img))  # 得到的假的图片与真实的图片的label的loss
                # bp and optimize
                R_optimizer.zero_grad()  # 梯度归0
                R_loss.backward()  # 进行反向传播
                R_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

                # 打印中间的损失
                if (i + 1) % 200 == 0:

                    print('\rEpoch[{}/{}],R_loss:{:.6f} '.format(
                        epoch, epochs, R_loss.mean().item()  # 打印的是真实图片的损失均值
                    ), end=' ')
            if (epoch+1)%200 == 0:
                from utils.utils import checkPath
                checkPath(self.src_RAN_model)
                torch.save(R.state_dict(), "%s/R_model1002-%d.pth" % (self.src_RAN_model,epoch))
                #torch.save(D.state_dict(), "%s/D_model0910-%d.pth" % (self.src_RAN_model,epoch))

    @showTime
    def train_old(self, epochs=10001, batch_size=128, sample_interval=500):
        # 数据加载
        dataSet = GetDataset(0, self.data_cols, self.latent_dim, self.batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        writer = SummaryWriter('./runs/RANs')
        # 创建对象
        D = discriminator(self.latent_dim)
        R = reinventor(self.latent_dim)
        D = D.to(device)
        R = R.to(device)
        '''if torch.cuda.is_available():
            D = D.cuda()
            R = R.cuda()'''

        # 首先需要定义loss的度量方式  （二分类的交叉熵）
        # 其次定义 优化函数,优化函数的学习率为0.0003
        criterion = nn.BCELoss()  # BCELoss()  # 交叉熵损失函数
        # criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.SmoothL1Loss()

        d_optimizer = torch.optim.SGD(D.parameters(), lr=0.002, momentum=0.9)#Adam(D.parameters(), lr=0.0005)#
        R_optimizer = torch.optim.Adam(R.parameters(),lr=0.002)  # Adam(R.parameters(),0.001)SGD(R.parameters(), lr=0.002, momentum=0.9)#

        real_label = Variable(torch.ones(batch_size)).to(device) # 定义真实的图片label为1
        fake_label = Variable(torch.zeros(batch_size)).to(device)  # 定义假的图片的label为0


        # ##########################进入训练##判别器的判断过程#####################
        for epoch in range(epochs):  # 进行多个epoch的训练
            for i in range(1000):

                # view()函数作用是将一个多行的Tensor,拼接成一行
                # 第一个参数是要拼接的tensor,第二个参数是-1
                # =============================训练判别器==================

                if i % 50 == 0:
                    target, input_ = dataSet.getNxetDataset()
                    target = dataSet.Normalize(target, 1)
                    real_img = Variable(target.type(torch.float32)).to(device)  # 将tensor变成Variable放入计算图中


                z = Variable(input_.type(torch.float32)).to(device)  # 随机生成一些噪声
                fake_img = R(z).detach()  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
                # fake_img = dataSet.Normalize(fake_img, 0)
                # print('mean', fake_img.mean().item(),'min', fake_img.min().item(),'max', fake_img.max().item())
                fake_img = dataSet.Normalize(fake_img, 3) # 3:不保存参数

                # ########判别器训练train#####################
                # 分为两部分：1、真的图像判别为真；2、假的图像判别为假

                # 计算真实图片的损失
                real_out = D(real_img)  # 将真实图片放入判别器中
                real_out = real_out.squeeze()  # (128,1) -> (128,)
                d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
                real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好

                # 计算假的图片的损失
                z = Variable(input_.type(torch.float32)).to(device)  # 随机生成一些噪声
                fake_img = R(z).detach()  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
                fake_out = D(fake_img)  # 判别器判断假的图片，
                #print('fake_out.size',fake_out.size())
                fake_out = fake_out.squeeze()  # (128,1) -> (128,)
                d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
                #print('d_loss_fake.mean', d_loss_fake.mean())
                #print('fake_out and fake_scores.mean', fake_out.mean())
                fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
                # 损失函数和优化
                d_loss = (d_loss_real + d_loss_fake) /2 # 损失包括判真损失和判假损失
                #print('d_loss.mean', d_loss.mean())
                #sys.exit()
                d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
                d_loss.backward()  # 将误差反向传播
                d_optimizer.step()  # 更新参数

                # ==================训练生成器============================
                # ###############################生成网络的训练###############################
                # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
                # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
                # 反向传播更新的参数是生成网络里面的参数，
                # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
                # 这样就达到了对抗的目的
                # 计算假的图片的损失
                z = Variable(input_.type(torch.float32)).to(device)  # 得到随机噪声
                fake_img = R(z)  # 随机噪声输入到生成器中，得到一副假的图片
                output = D(fake_img.type(torch.float32))  # 经过判别器得到的结果
                output = output.squeeze()
                R_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss
                # bp and optimize
                R_optimizer.zero_grad()  # 梯度归0
                R_loss.backward()  # 进行反向传播
                R_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

                # 打印中间的损失
                if (i + 1) % 200 == 0:
                    print('\rEpoch[{}/{}],d_loss:{:.6f},r_loss:{:.6f} '
                          'D real: {:.6f},D fake: {:.6f}'.format(
                        epoch, epochs, d_loss.mean().item(), R_loss.mean().item(),
                        real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
                    ),end=' ')
                '''if epoch == 0:
                    real_images = to_img(real_img.cpu().data)
                    save_image(real_images, './img/real_images.png')'''
            #fake_images = to_img(fake_img.cpu().data)
            #save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))

            #writer.add_scalar('d_loss', d_loss.data.item(), epoch)
            #writer.add_scalar('R_loss', R_loss.data.item(), epoch)
            #writer.close()

            if (epoch+1)%400 == 0:
                dataSet.renewData()
            if (epoch+1)%500 == 0:
                from utils.utils import checkPath
                checkPath(self.src_RAN_model)
                torch.save(R.state_dict(), "%s/R_model0910-%d.pth" % (self.src_RAN_model,epoch))
                #torch.save(D.state_dict(), "%s/D_model0910-%d.pth" % (self.src_RAN_model,epoch))


    def test(self, state_num=2499 ,flag = False,test_state = "test.xlsx"):
        # 数据加载
        dataSet = GetDataset(0, self.data_cols, self.latent_dim, self.batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src_Rein_model = "%s/R_model0910-%d.pth" % (self.src_RAN_model, state_num)

        # 创建改造对象
        R = reinventor(self.latent_dim)
        R.load_state_dict(torch.load(src_Rein_model))
        R = R.to(device)

        # 获取
        testDatas, base = dataSet.getTestData()
        base_np = np.array(base)
        base_np = base_np.reshape(base_np.shape[0],base_np.shape[-1])
        #print(testDatas[0].shape)
        #print(base_np.shape)

        idy = 0
        flagRenew = False
        for testData in testDatas:
            z = Variable(testData.type(torch.float32)).to(device)  # 随机生成一些噪声
            fake_img = R(z).detach()
            #print(fake_img.shape)
            fake_img = np.array(fake_img)
            fake_img = fake_img.reshape(fake_img.shape[0], fake_img.shape[-1])
            #print(fake_img.shape)
            base_np[:,idy : idy + self.latent_dim] = fake_img * (dataSet.testDataMax)*30
            idy += self.latent_dim
            flagRenew = True

        # 返回测试数组？默认不返回
        if flagRenew:
            if not os.path.exists(self.src_RAN_test):
                os.makedirs(self.src_RAN_test)
            # 写表格
            self.data_write_xlsx(666, base_np, )
            if flag :
                return base_np


    def data_write_xlsx(self, epoch, gen_datas, num=256):
        if not os.path.exists(self.src_RAN_data):
            os.makedirs(self.src_RAN_data)
        if epoch == 666:
            file_name = "%s/test.xlsx"%self.src_RAN_test
        else:
            file_name = "%s/%d.xlsx" % (self.src_RAN_data,epoch)
        print(file_name)
        gen_datas = gen_datas.reshape(num, self.data_cols)
        dt = pd.DataFrame(gen_datas)
        dt.to_excel(file_name, index=0)

    def forecast(self,dataName,state_num=4999,state = -1,outputName = 'outputName'):
        '''
        从彩色图像SR彩色图像
        '''
        import utils.utils as zxhUtils
        if isinstance(dataName, str):
            if dataName[-4] != '.':
                dataName = dataName + ".jpg"
            input_RGBs = zxhUtils.getRGB(dataName)
            gray = cv2.merge((input_RGBs[0],input_RGBs[1],input_RGBs[2]))
            merge_RGBs =[]
            for input_RGB in input_RGBs:
                merge_RGBs.append(self.foreCast(input_RGB, state_num, state, 'return'))
            merge_img = merge_RGBs[0]
            if len(merge_RGBs)==3:
                # print('merge_RGBs[0]',merge_RGBs[0][0][2])
                merge_img = cv2.merge((merge_RGBs[0],merge_RGBs[1],merge_RGBs[2]))
            if not dataName == 'outputName':

                pasth = self.src_RAN_data + '/' + resTime()[0:-4]
                zxhUtils.checkPath(pasth)

                cv2.imwrite(pasth + '/origin.jpg', gray)
                cv2.imwrite(pasth + '/' + outputName + '.jpg', np.array(merge_img))




    def foreCast(self,dataName,state_num=4999,state = -1,outputName = 'outputName'):
        '''
        返回灰度SR图像
        outputName = ’return‘时返回重建的单通道值或灰度值
        '''
        import utils.utils as zxhUtils

        gray,inData,Target = zxhUtils.getTargetWavelets(dataName, state ,True)
        #inData =torch.from_numpy(np.array(inData))
        inData = np.array(inData)

        # 数据加载
        dataSet = GetDataset(0, self.data_cols, self.latent_dim, self.batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src_Rein_model = "%s/R_model1002-%d.pth" % (self.src_RAN_model, state_num)#R_model0910-%d.pth" % (self.src_RAN_model, state_num)

        # 创建改造对象
        R = reinventor(self.latent_dim)
        R.load_state_dict(torch.load(src_Rein_model))
        R = R.to(device)

        # 获取
        testDatas, base = dataSet.getTestData(False,inData)
        base_np = np.array(base)
        base_np = base_np.reshape(base_np.shape[0], base_np.shape[-1])

        idy = 0
        #flagRenew = False
        for testData in testDatas:
            z = Variable(testData.type(torch.float32)).to(device)  # 随机生成一些噪声
            fake_img = R(z).detach()
            # print(fake_img.shape)
            fake_img = np.array(fake_img)
            fake_img = fake_img.reshape(fake_img.shape[0], fake_img.shape[-1])
            # print(fake_img.shape)
            # print('z.shape',z.shape)
            base_np[:, idy: idy + self.latent_dim] = fake_img  * (dataSet.testDataMax) * 10* abs(np.array(z).reshape(fake_img.shape[0], fake_img.shape[-1]))
            idy += self.latent_dim
            #flagRenew = True

        #base_np.tolist()
        targetImg = np.array(zxhUtils.resImgFormWavelat(Target,base_np,state)).astype(np.uint8)
        if outputName == 'return':
            return targetImg
        import cv2
        pasth =self.src_RAN_data+'/'+resTime()[0:-4]
        zxhUtils.checkPath(pasth)
        cv2.imwrite('outputName.jpg', np.array((targetImg)))
        cv2.imwrite(pasth+'/'+outputName+'.jpg',np.array(targetImg))
        if not dataName == 'outputName':
            cv2.imwrite(pasth + '/origin.jpg', gray)
        #zxhUtils.getPlot(gray,targetImg)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_targetData',type=str, default='dataSet/targetData/targetData_level-2_0.xlsx', help='目标样本地址')
    parser.add_argument('--src_noiseData', type=str, default='dataSet/inputData/inputData_level-2_0.xlsx', help='带改造样本地址')
    parser.add_argument('--src_testData', type=str, default='dataSet/testData/test.xlsx', help='测试样本地址')
    parser.add_argument('--data_Number', type=int, default=1563, help='数据表总量')
    
    parser.add_argument('--data_rows', type=int, default=1, help='数据行数')
    parser.add_argument('--data_cols', type=int, default=123, help='数据列数')
    parser.add_argument('--channels', type=int, default=1, help='数据通道数')
    parser.add_argument('--latent_dim', type=int, default=20, help='生成器输入大小')
    parser.add_argument('--sample_size', type=int, default=1280, help='样本量大小')
    parser.add_argument('--sampleSize', type=int, default=256, help='一张图像样本量大小')
    parser.add_argument('--count', type=int, default=100, help='量化大小')
    parser.add_argument('--count_noise', type=int, default=50, help='带改造样本量化大小')
    
    parser.add_argument('--epochs', type=int, default=10001, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=128, help='每轮训练数据量(类似图片量)')
    parser.add_argument('--sample_lenth', type=int, default=20, help='单个训练数据')
    parser.add_argument('--sample_interval', type=int, default=500, help='输出数据间隔')
    opt = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(opt.src_testData)
    #ran = RAN(opt)
    #print(ran.getSempleInterval())
    get=GetDataset(1)
    #compressImageRes("../im4.jpg","000.jpg",2,0,0)
    '''ran = RAN()
    ran.train(epochs=20001, batch_size=16, sample_interval=500)
    ran.test()'''

