#import glob
#import logging
#import math
import os
import cv2
import torch
import pywt

def checkPath(path,flag = False):
    """
    Parameters
    ----------
    检查地址是否存在
    不存在就生成
    默认关闭提醒和返回
    Returns
    -------
    path : 地址

    """
    # Search for file if not found
    
    if os.path.exists(path) or path == '':
        if flag:
            print("already existed!")
            return path
    else:
        os.makedirs(path)
        if flag:
            print("create success!")
            return path

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
    if (level_set == 0):
        return coffs
    return coffs[level_set]

def getGray(fileName,output = False):
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

def getRGB(fileName):
    '''
    从文件名返回三通道图像
    '''
    img = cv2.imread(fileName)

    return img[:,:,0] , img[:,:,1] , img[:,:,2]

def getTargetWavelets(input_name,waveletLevel=-2,retrunGray = False):
    inData = []
    Target = []
    if isinstance(input_name,str):
        if input_name[-4]!='.':
            input_name =input_name +".jpg"
        input_gray = getGray(input_name)
    else:
        input_gray = input_name
    # target_gray = get_gray(target_name)
    for count_num in range(len(input_gray)):
        inData.append(test_get_a_wavelet(input_gray[count_num], waveletLevel))
        Target.append(test_get_a_wavelet(input_gray[count_num], 0))
    if retrunGray:
        return input_gray , inData, Target
    return Target,inData

def getTargetWavelet(input_name,waveletLevel=-2):

    input_gray = get_gray(input_name)
    #target_gray = get_gray(target_name)
    for count_num in range(len(input_gray)):
        inData.append(test_get_a_wavelet(input_gray[count_num], waveletLevel))
        #targetData.append(test_get_a_wavelet(target_gray[count_num], -2))

def resImgFormWavelat(base,targetWave,Level=-2):
    if not isinstance(base,list):
        base.tolist()
    if not isinstance(targetWave,list):
        targetWave.tolist()
    baseImg =[]
    for count in range(len(base)):
        base[count][Level]=targetWave[count]

        baseImg.append(pywt.waverec(base[count], 'db8'))  # 将信号进行小波重构

    return baseImg

# 绘图函数
def getPlot(input_, target):
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(input_, cmap='gray')
    plt.title('origin')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(target, cmap='gray')
    plt.title('result')
    plt.axis('off')
    plt.show()

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


# 放大res_num倍，然后并输出，改变大小
def overSizeImageRes(srcFile, dstFile, res_num: int = 2, flag_out=True, flag_return= False):
    """
    Parameters
    ----------
    srcFile : 输入图像，可以是文件名地址也可以是已经读取了的CV2图像
    dstFile : 输出地址，不输出也需要占位置
    res_num : int 放缩倍数
    flag_out : 输出文件的标志
    flag_return : 返回图像的标志

    Returns
    -------
    返回CV2格式的图像

    """
    dstFile = dstFile[0:-4] + '_overSizelevel' + str(res_num) + dstFile[-4:]
    import PIL.Image as Image
    import os
    import numpy
    import cv2
    if os.path.isfile(srcFile):
        try:
            # 打开原图片缩小后保存，可以用if srcFile.endswith(".jpg")或者split，splitext等函数等针对特定文件压缩
            sImg = Image.open(srcFile)
            w, h = sImg.size
            dImg = sImg.resize((int(w * res_num), int(h * res_num)), Image.ANTIALIAS)  # 设置压缩尺寸和选项，注意尺寸要用括号
            res_dImg = dImg#dImg.resize((w, h), Image.ANTIALIAS)  # 设置压缩尺寸和选项，注意尺寸要用括号
            if flag_out:
                res_dImg.save(dstFile)
            if flag_return:
                return cv2.cvtColor(numpy.asarray(res_dImg), cv2.COLOR_RGB2BGR)  # PIL Image 转 cv2
            # res_dImg.save(dstFile) #也可以用srcFile原路径保存,或者更改后缀保存，save这个函数后面可以加压缩编码选项JPEG之类的
            print(dstFile + " 成功！")
            # return dstFile
        except Exception:
            print(dstFile + "失败！！！！！！！！！！！！")
    else:
        # cv2模式已经读取的图像

        sImg = Image.fromarray(cv2.cvtColor(srcFile, cv2.COLOR_BGR2RGB))  # cv2转PIL Image
        w, h = sImg.size
        dImg = sImg.resize((int(w * res_num), int(h * res_num)), Image.ANTIALIAS)  # 设置压缩尺寸和选项，注意尺寸要用括号
        res_dImg = dImg.resize((w, h), Image.ANTIALIAS)  # 设置压缩尺寸和选项，注意尺寸要用括号
        if flag_out:
            res_dImg.save(dstFile)
        if flag_return:
            return cv2.cvtColor(numpy.asarray(res_dImg), cv2.COLOR_RGB2BGR)  # PIL Image 转 cv2

#放缩res_num倍，然后放缩回来，即只改变清晰度，不改变大小
def compressImageRes(srcFile,dstFile ,res_num:int,flag_out = False,flag_return = True):
    """
    Parameters
    ----------
    srcFile : 输入图像，可以是文件名地址也可以是已经读取了的CV2图像
    dstFile : 输出地址，不输出也需要占位置
    res_num : int 放缩倍数
    flag_out : 输出文件的标志
    flag_return : 返回图像的标志

    Returns
    -------
    返回CV2格式的图像

    """
    dstFile = dstFile[0:-4]+'_level'+str(res_num)+dstFile[-4:]
    import PIL.Image as Image
    import os
    import numpy
    if os.path.isfile(srcFile):
        try:
            #打开原图片缩小后保存，可以用if srcFile.endswith(".jpg")或者split，splitext等函数等针对特定文件压缩
            sImg=Image.open(srcFile)
            w,h=sImg.size
            dImg=sImg.resize((int(w/res_num),int(h/res_num)),Image.ANTIALIAS)  #设置压缩尺寸和选项，注意尺寸要用括号
            res_dImg=dImg.resize((w,h),Image.ANTIALIAS)  #设置压缩尺寸和选项，注意尺寸要用括号
            if flag_out:
                res_dImg.save(dstFile)
            if flag_return:
                return cv2.cvtColor(numpy.asarray(res_dImg),cv2.COLOR_RGB2BGR)# PIL Image 转 cv2
            #res_dImg.save(dstFile) #也可以用srcFile原路径保存,或者更改后缀保存，save这个函数后面可以加压缩编码选项JPEG之类的
            print (dstFile+" 成功！")
            #return dstFile
        except Exception:
            print(dstFile+"失败！！！！！！！！！！！！")
    else:
        #cv2模式已经读取的图像
        
        sImg= Image.fromarray(cv2.cvtColor(srcFile,cv2.COLOR_BGR2RGB))#cv2转PIL Image
        w,h=sImg.size
        dImg=sImg.resize((int(w/res_num),int(h/res_num)),Image.ANTIALIAS)  #设置压缩尺寸和选项，注意尺寸要用括号
        res_dImg=dImg.resize((w,h),Image.ANTIALIAS)  #设置压缩尺寸和选项，注意尺寸要用括号
        if flag_out:
            res_dImg.save(dstFile)
        if flag_return:
            return cv2.cvtColor(numpy.asarray(res_dImg),cv2.COLOR_RGB2BGR)# PIL Image 转 cv2
        
def randomColors():
    import random
    colors=['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
    #colors=["blue","black","brown","red","green","orange"   ,"yellow","beige","turquoise","pink"]
    random.shuffle(colors)
    #print(colors)
    return colors[0]

if __name__ == '__main__':
    pass
    compressImageRes("im4.jpg","00000.jpg",2,1,0)
    
    '''
    compressImageRes(getGray("im4.jpg"),"00000.jpg",2,1,0)
    pass
    print(checkPath("dateSet/targetData/targetDat3",1))'''
    