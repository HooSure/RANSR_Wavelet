import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import log
from PIL import Image
import datetime
import pywt
import os

# 以下强行用Python宏定义变量
halfWindowSize = 9


src1_path = 'G:/Desktop_ZXH/Vimeo-90k-Septuplet dataset/input'
src2_path = 'G:/Desktop_ZXH/Vimeo-90k-Septuplet dataset/target'


def fcode(src1_path,src2_path):
    '''
    来自敬忠良，肖刚，李振华《图像融合——理论与分析》P85：基于像素清晰度的融合规则
    1，用Laplace金字塔或者是小波变换，将图像分解成高频部分和低频部分两个图像矩阵
    2，以某个像素点为中心开窗，该像素点的清晰度定义为窗口所有点((高频/低频)**2).sum()
    3，目前感觉主要的问题在于低频
    4，高频取清晰度图像中较大的那个图的高频图像像素点
    5，算法优化后速度由原来的2min.44s.变成9s.305ms.
    补充：书上建议开窗大小10*10，DWT取3层，Laplace金字塔取2层
    '''
    

    def imgOpen(img_src1, img_src2):
        apple = Image.open(img_src1).convert('L')
        orange = Image.open(img_src2).convert('L')
        appleArray = np.array(apple)
        orangeArray = np.array(orange)
        return appleArray, orangeArray


    # 严格的变换尺寸
    def _sameSize(img_std, img_cvt):
        x, y = img_std.shape
        print('xy',x,y)
        pic_cvt = Image.fromarray(img_cvt)
        pic_cvt.resize((x, y))
        return np.array(pic_cvt)


    # 小波变换的层数不能太高，Image模块的resize不能变换太小的矩阵，不相同大小的矩阵在计算对比度时会数组越界
    def getWaveImg(apple, orange):
        appleWave = pywt.wavedec2(apple, 'haar', level=4)
        orangeWave = pywt.wavedec2(orange, 'haar', level=4)

        lowApple = appleWave[0];
        lowOrange = orangeWave[0]
        # 以下处理低频
        lowAppleWeight, lowOrangeWeight = getVarianceWeight(lowApple, lowOrange)
        lowFusion = lowAppleWeight * lowApple + lowOrangeWeight * lowOrange
        # 以下处理高频
        for hi in range(1, 5):
            waveRec = []
            for highApple, highOrange in zip(appleWave[hi], orangeWave[hi]):
                highFusion = np.zeros(highApple.shape)
                #print('0',hi)
                contrastApple = getContrastImg(lowApple, highApple)
                contrastOrange = getContrastImg(lowOrange, highOrange)
                row, col = highApple.shape
                for i in range(row):
                    for j in range(col):
                        if contrastApple[i, j] > contrastOrange[i, j]:
                            highFusion[i, j] = highApple[i, j]
                        else:
                            highFusion[i, j] = highOrange[i, j]
                waveRec.append(highFusion)
            recwave = (lowFusion, tuple(waveRec))
            lowFusion = pywt.idwt2(recwave, 'haar')
            lowApple = lowFusion;
            lowOrange = lowFusion
        return lowFusion


    # 求Laplace金字塔
    def getLaplacePyr(img):
        firstLevel = img.copy()
        secondLevel = cv2.pyrDown(firstLevel)
        lowFreq = cv2.pyrUp(secondLevel)
        highFreq = cv2.subtract(firstLevel, _sameSize(firstLevel, lowFreq))
        return lowFreq, highFreq


    # 计算对比度，优化后不需要这个函数了，扔在这里看看公式就行
    def _getContrastValue(highWin, lowWin):
        row, col = highWin.shape
        contrastValue = 0.00
        for i in range(row):
            for j in range(col):
                contrastValue += (float(highWin[i, j]) / lowWin[i, j]) ** 2
        return contrastValue


    # 先求出每个点的(hi/lo)**2，再用numpy的sum（C语言库）求和
    def getContrastImg(low, high):
        row, col = low.shape
        if low.shape != high.shape:
            print(len(high),len(low))
            low = _sameSize(high, low)
            print(len(high),len(low))
        contrastImg = np.zeros((row, col))
        
        contrastVal = (high / low) ** 2
        for i in range(row):
            for j in range(col):
                up = i - halfWindowSize if i - halfWindowSize > 0 else 0
                down = i + halfWindowSize if i + halfWindowSize < row else row
                left = j - halfWindowSize if j - halfWindowSize > 0 else 0
                right = j + halfWindowSize if j + halfWindowSize < col else col
                contrastWindow = contrastVal[up:down, left:right]
                contrastImg[i, j] = contrastWindow.sum()
        return contrastImg


    # 计算方差权重比
    def getVarianceWeight(apple, orange):
        appleMean, appleVar = cv2.meanStdDev(apple)
        orangeMean, orangeVar = cv2.meanStdDev(orange)
        appleWeight = float(appleVar) / (appleVar + orangeVar)
        orangeWeight = float(orangeVar) / (appleVar + orangeVar)
        return appleWeight, orangeWeight


    # 函数返回融合后的图像矩阵
    def getPyrFusion(apple, orange):
        lowApple, highApple = getLaplacePyr(apple)
        lowOrange, highOrange = getLaplacePyr(orange)
        contrastApple = getContrastImg(lowApple, highApple)
        contrastOrange = getContrastImg(lowOrange, highOrange)
        row, col = lowApple.shape
        highFusion = np.zeros((row, col))
        lowFusion = np.zeros((row, col))
        # 开始处理低频
        # appleWeight,orangeWeight=getVarianceWeight(lowApple,lowOrange)
        for i in range(row):
            for j in range(col):
                # lowFusion[i,j]=lowApple[i,j]*appleWeight+lowOrange[i,j]*orangeWeight
                lowFusion[i, j] = lowApple[i, j] if lowApple[i, j] < lowOrange[i, j] else lowOrange[i, j]
        # 开始处理高频
        for i in range(row):
            for j in range(col):
                highFusion[i, j] = highApple[i, j] if contrastApple[i, j] > contrastOrange[i, j] else highOrange[i, j]
        # 开始重建
        fusionResult = cv2.add(highFusion, lowFusion)
        return fusionResult


    # 绘图函数
    def getPlot(apple, orange, result):
        plt.subplot(131)
        plt.imshow(apple, cmap='gray')
        plt.title('src1')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(orange, cmap='gray')
        plt.title('src2')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(result, cmap='gray')
        plt.title('result')
        plt.axis('off')
        plt.show()


    # 画四张图的函数，为了方便同时比较
    def cmpPlot(apple, orange, wave, pyr):
        plt.subplot(221)
        plt.imshow(apple, cmap='gray')
        plt.title('SRC1')
        plt.axis('off')
        plt.subplot(222)
        plt.imshow(orange, cmap='gray')
        plt.title('SRC2')
        plt.axis('off')
        plt.subplot(223)
        plt.imshow(wave, cmap='gray')
        plt.title('WAVELET')
        plt.axis('off')
        plt.subplot(224)
        plt.imshow(pyr, cmap='gray')
        plt.title('LAPLACE PYR')
        plt.axis('off')
        plt.show()


    def runTest(src1=src1_path, src2=src2_path, isplot=True):
        apple, orange = imgOpen(src1, src2)
        beginTime = datetime.datetime.now()
        print(beginTime)
        #拉普拉斯变换
        pyrResult = getPyrFusion(apple, orange)
        #小波变换
        waveResult = getWaveImg(apple, orange)
        return
        endTime = datetime.datetime.now()
        print(endTime)
        print('Runtime: ' + str(endTime - beginTime))
        if isplot:
            cmpPlot(apple, orange, waveResult, pyrResult)
        return waveResult, pyrResult
    
    
    src1_path = src1_path + '.jpg'
    src2_path = src2_path + '.jpg'
    runTest(src1_path,src2_path)

    
def test_gray(fileName,output = True):
    input_name = fileName + '.jpg'
    img = cv2.imread(input_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if output:
        output_name = fileName + '_gray.jpg'
        cv2.imwrite(output_name,gray)
    return gray
def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):    
        print(dirs) 
        break


#-------------------------------------------------------------------#
#                  以下为获取Vimeo-90k的程序                         #
#                                                                   #

def get_images(isr = src2_path):
    from tqdm import tqdm
    '''
    获取所有图片
    '''
    inputs,targets,lenth = get_images_paths(isr)
    
    #print(lenth)
    if lenth>100:
        inDataPath = 'excel/inputData/inputData.xlsx'
        targetDataPath = 'excel/targetData/targetData.xlsx'
        
        def next_excel(num,jiange,lenth, inDataPath = 'excel/inputData/inputData.xlsx',targetDataPath = 'excel/targetData/targetData.xlsx'):
            '''
            ----------
            num : '次数'
            jiange : '间隔'
            lenth : 总路径数
            inDataPath : 低清数据存放路径蓝本
                DESCRIPTION. The default is 'excel/inputData/inputData.xlsx'.
            targetDataPath : 高清数据存放路径蓝本
                DESCRIPTION. The default is 'excel/targetData/targetData.xlsx'.
            '''
            inData = []
            targetData = []
            for count in range(num*jiange , (num+1)*jiange):
                input_name = inputs[count]
                target_name = targets[count]
                #print(input_name)
                print("\r获取数据：{}/{}".format(count,lenth), end="")
                input_gray = get_gray(input_name)
                target_gray = get_gray(target_name)
                for count_num in range(len(input_gray)):
                    inData.append(test_get_a_wavelet(input_gray[count_num],-2))
                    targetData.append(test_get_a_wavelet(target_gray[count_num],-2))
            print("获取数据完成！")
                
            inDataPath = inDataPath[:-5]+'_level-2_'+str(num)+inDataPath[-5:]
            targetDataPath = targetDataPath[:-5]+'_level-2_'+str(num)+targetDataPath[-5:]
            print("开始写表1！")
            draw_a_excel(inData,inDataPath)
            print("开始写表2！")
            draw_a_excel(targetData,targetDataPath)
            print('Fnish!')
        for nn in tqdm(range(int(lenth/5))):
            next_excel(nn,5,lenth)

def get_gray(fileName,output = False):
    '''
    从文件名返回灰度图像
    '''
    #fileName = fileName.replace('\\','/')
    img = cv2.imread(fileName)
    #print(fileName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if output:
        output_name = fileName[-25:-4] + '_gray.jpg'
        cv2.imwrite(output_name,gray)
    return gray

def get_images_paths(isr = src2_path):
    '''
    获取所有图片地址
    '''
    test_isrs = getAllFiles(isr)
    input_file_path_list = []
    target_file_path_list = []
    for path in test_isrs:
        input_file,target_file = get_input(path)
        if input_file != 'error':
            input_file_path_list.append(input_file)
            target_file_path_list.append(target_file)
        if len(input_file_path_list)!=len(target_file_path_list):
            print('数据不对等！')
            return 'error lenth'
    return input_file_path_list,target_file_path_list,len(input_file_path_list)
    
    
def getAllFiles(folder):
    '''
    返回所有目录下及子目录下的文件
    '''
    filepath_list = []
    for root,folder_names, file_names in os.walk(folder):
        for file_name in file_names:
            file_path = root + os.sep + file_name
            filepath_list.append(file_path)
            #print(file_path)
    file_path = sorted(file_path, key=str.lower)
    return filepath_list

def get_input(src):
    '''
    input 对应的 target
    返回两个地址
    '''
    if src!='':
        src_test = src[-7:]
        if src_test == 'im4.png' and src[-25:-19] == 'target':
            #print(src[-25:-19])
            return src[:-25]+'input'+src[-19:],src
        else:
            print('出现错误，错误地址：',src)
            return 'error','error',
        
        
def test_get_a_wavelet(data, level_set :int = -1):
    '''
    从一维数据返回任意一列小波系数
    '''
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)

    if maxlev < level_set:
        level_set = maxlev
    # level_set -= 1
    coffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解
    return coffs[level_set]

def draw_a_excel(input_Data, outpath_excel):
    from openpyxl import Workbook
    
    def draw_excel(in_Data):
        
        # 打开表格工作区
        wb = Workbook()
        ws = wb.active  # 激活 worksheet
        num = 0
        lenth = len(in_Data)
        for Data in in_Data:
            num = num+1
            Data = Data.tolist()
            #print(Data.shape)
            print("\r填写数据：{}/{}".format(num,lenth), end="")
            ws.append(Data)
            '''row = []
            for iData in Data:
                row.append(iData)
            ws.append(row)'''

        wb.save(outpath_excel)
        print("填写完毕！")

    if ((len(input_Data)) > 1):
        draw_excel(input_Data)
    else:
        print("填写数据有误！")
        return "errer"


if __name__ == '__main__':
    #fcode('001', '001_res_level2')
    #test_gray('001_res_level2')
    #gray = test_gray('001',False)
    #get_images()
    get_images()
    #print(get_input(getAllFiles(src2_path)))
