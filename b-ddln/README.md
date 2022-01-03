# 运用B-DDLN诊断COVID-19
1. 所有代码文件要在英文路径下运行；
2. 数据集获取方式
链接：https://pan.baidu.com/s/1mz6gVSX0m9EeGuZbbF9V4Q 
提取码：gpyx
3. 数据集在Dataset文件夹里，分为两类：正常（Normal）和COVID-19，Dataset/VERIFICATION是测试集图像；
4. 用GPU运行FeatureExtractor.py代码文件，等特征提取器模型训练完成后输入训练图像和测试图像得到各自的512维特征并存储起来；
5. 利用存储好的训练图像特征训练集成动态学习网络分类器，并运用测试图像特征进行模型评估。
