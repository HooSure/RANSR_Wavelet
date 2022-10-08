from __future__ import print_function, division
import os
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import pandas as pd
#import matplotlib.pyplot as plt
#import sys
import numpy as np
##import xlrd
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class RAN():
    def __init__(self):
        self.data_rows = 1
        self.data_cols = 123
        self.channels = 1
        self.data_shape = (self.data_rows, self.data_cols)
        self.latent_dim = 123
        self.sample_size = 1280
        self.sampleSize = 256

        self.count = 150
        self.count_noise= 50
        
        self.src_targetData= "dataSet/targetData/targetData_level-2_1.xlsx"
        self.src_noiseData= "dataSet/inputData/inputData_level-2_1.xlsx"
        self.src_testData = "dataSet/testData/test.xlsx"
        
        optimizer = Adam(0.0002, 0.5)
        # 构建和编译判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.reinventor = self.build_reinventor()
        z = Input(shape=(self.latent_dim,))
        data = self.reinventor(z)  # 生成器生成的图片
        self.discriminator.trainable = True

        # 判别器将生成的图像作为输入并确定有效性
        validity = self.discriminator(data)  # 这个是判别器判断生成器生成图片的结果

        # 组合模型(叠加发生器和鉴别器)
        # 训练生成器骗过判别器
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_reinventor(self):
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        '''model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))'''

        # np.prod(self.img_shape)=28x28x1
        model.add(Dense(np.prod(self.data_shape), activation='tanh'))
        model.add(Reshape(self.data_shape))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        data = model(noise)

        # 输入噪音，输出图片
        return Model(noise, data)

    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.data_shape))
        '''model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))'''
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        data = Input(shape=self.data_shape)
        validity = model(data)

        return Model(data, validity)
    #训练轮次 数据量大小  样本间距

    def train(self, epochs, batch_size=128, sample_interval=500):
        # 加载数据集
        data = pd.read_excel(self.src_target, header=None)
        # 加载改造集
        noise_data = pd.read_excel(self.src_noiseData, header=None)
        data = np.array(data.values.tolist()).reshape(self.sample_size, 1, self.latent_dim)
        noise_data = np.array(noise_data.values.tolist()).reshape(self.sample_size, 1, self.latent_dim)
        #print(data.shape)
        #print(noise_data.shape)
        # 将数据进行归一化处理
        data = data/self.count
        noise_data = noise_data/self.count_noise

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            # -----------------------------------------------------------------------------------------------------#
            #  训练判别器
            # ---------------------
            # X_train.shape[0]为数据集的数量，随机生成batch_size个数量的随机数，作为数据的索引
            idx = np.random.randint(0, data.shape[0], batch_size)
            # 从数据集随机挑选batch_size个数据，作为一个批次训练
            x = data[idx]
            # 噪音维度(batch_size,100)
            noise = noise_data[idx].reshape(batch_size, self.latent_dim)

            # 由改造器根据数据改造？这是啥一个朋友测试
            RAN_x = self.reinventor.predict(noise)

            # 训练判别器，判别器希望真实图片，打上标签1，假的图片打上标签0
            d_loss_real = self.discriminator.train_on_batch(x, valid)
            # print(d_loss_real)
            d_loss_fake = self.discriminator.train_on_batch(RAN_x, fake)
            # print(d_loss_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # print(d_loss)
            
            # -----------------------------------------------------------------------------------------------------#
            #  训练生成器
            # ---------------------
            idx = np.random.randint(0, data.shape[0], batch_size)
            noise = noise_data[idx].reshape(batch_size, self.latent_dim)
            # Train the generator (to have the discriminator label samples as valid)
            R_loss = self.combined.train_on_batch(noise, valid)
            # gen_x = (gen_x + 1) * 192
            # 打印loss值
            print("%d [D loss: %f, acc: %.2f%%] [R loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], R_loss))
            # print("data", gen_x)
            # 每sample_interval个epoch保存一次生成图片
            if epoch % sample_interval == 0:
                self.sample_data(epoch)
                if not os.path.exists("RAN_model"):#不存在就生成！
                    os.makedirs("RAN_model")
                self.reinventor.save_weights("RAN_model/R_model%d.hdf5" % epoch, True)
                self.discriminator.save_weights("RAN_model/D_model%d.hdf5" % epoch, True)
            if (epoch % 2000 == 0) and epoch != 0:
                num = random.randint(1,56)
                src_target_re = self.src_target[:-6]+str(num)+self.src_target[-5:]
                src_noiseData_re = self.src_noiseData[:-6]+str(num)+self.src_noiseData[-5:]
                # 加载数据集
                data = pd.read_excel(src_target_re, header=None)
                # 加载改造集
                noise_data = pd.read_excel(src_noiseData_re, header=None)

                data = np.array(data.values.tolist()).reshape(self.sample_size, 1, self.latent_dim)
                noise_data = np.array(noise_data.values.tolist()).reshape(self.sample_size, 1, self.latent_dim)

    def data_write_xlsx(self, epoch, gen_datas, num=256):
        if not os.path.exists("RAN_data"):
            os.makedirs("RAN_data")
        if epoch == 666:
            file_name = "RAN_test/test.xlsx"
        else:
            file_name = "RAN_data/%d.xlsx" % epoch
        print(file_name)
        gen_datas = gen_datas.reshape(num, self.data_cols)
        dt = pd.DataFrame(gen_datas)
        dt.to_excel(file_name, index=0)
        # with open(file_name, "w", encoding="utf-8", newline'') as f:
        #     writer = csv.writer(f)
        #     for data in datas:
        #         writer.writerow(data)
        #     print("保存文件成功，处理结束")

    def sample_data(self, epoch):
        # 重新生成一批噪音，维度为(self.sample_size,100)
        noise_data = pd.read_excel(self.testData, header=None)
        noise_data = noise_data / self.count_noise
        ran_datas = self.reinventor.predict(noise_data)
        ran_datas = ran_datas * self.count
        self.data_write_xlsx(epoch, ran_datas, self.sampleSize)

    def test(self, gen_nums=256):
        self.reinventor.load_weights("RAN_model/D_model5500.hdf5", by_name=True)
        self.discriminator.load_weights("RAN_model/D_model5500.hdf5", by_name=True)
        noise_data = pd.read_excel(self.testData, header=None)
        noise_data = noise_data / self.count_noise
        ran_datas = self.reinventor.predict(noise_data)
        ran_datas = ran_datas*self.count
        print(ran_datas)
        if not os.path.exists("RAN_test"):
            os.makedirs("RAN_test")
        self.data_write_xlsx(666, ran_datas, gen_nums)


if __name__ == '__main__':
    ran = RAN()
    ran.train(epochs=20001, batch_size=16, sample_interval=500)
    ran.test()

