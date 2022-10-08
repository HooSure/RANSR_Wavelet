# 项目名称：RANSR_Wavelet
##   总体描述 
### 1.功能介绍
1.图像SR

2.高频小波（道波西小波 - db8）
### 2.函数入口
train.py
```python
    opts = reOpts()
    ran = ModelRAN.RAN(opts)
    zxhUtils.overSizeImageRes('img_021_SRF_2_HR.jpg','test.jpg',2)
    ran.forecast('test_overSizelevel2',2599,-1,'outputName001')
```
### 3.注意事项
1.灰度图像重构：ran.foreCast()

2.彩色三通道重构：ran.forecast()
### 版本信息
- 最后更新时间：2022年10月08日
- 联系邮箱：Tbzzzxh@163.com
- 来自张小虎
