# 项目名称：RANSR_Wavelet
##   总体描述 
### 1.功能介绍
1.图像SR

2.高频小波（道波西小波 - db8）
### 2.函数入口
train.py
  [示例代码]  
```python
    opts = reOpts()
    ran = ModelRAN.RAN(opts)
    zxhUtils.overSizeImageRes('img_021_SRF_2_HR.jpg','test.jpg',2)
    ran.forecast('test_overSizelevel2',2599,-1,'outputName001')
```
### 3.注意事项
1.数据集地址
 [谷歌路径](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD)
 [百度网盘路径](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot#list/path=%2F)
 
2.灰度图像重构：ran.foreCast()

3.彩色三通道重构：ran.forecast()
### 版本信息
- 最后更新时间：2022年10月08日
- 联系邮箱：Tbzzzxh@163.com
- 来自张小虎
