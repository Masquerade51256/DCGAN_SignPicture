## DCGAN介绍

***UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS*** https://arxiv.org/pdf/1511.06434.pdf

***译文：***https://ask.julyedu.com/question/7681

## 项目描述及需求

#### **DCGAN**

生成对抗网络作为无监督学习的一种方法，网络包括一个生成模型G和一个判别模型D：

生成模型的输入是一个随机生成的向量，长度不定，输出是一个具有一定大小的图像。

判别模型的输入维度和G的输出一样，通过输出判定生成图片与真实图片是否相似。

通过两个模型的对抗过程，期望G生成一个足够真的图像，而D能保证验证的准确率。

## 项目准备

#### **帖子**

论文解析：
https://www.cnblogs.com/lyrichu/p/9054704.html

样例解析：
https://www.cnblogs.com/lyrichu/p/9093411.html

用DCGAN生成女朋友：
https://zhuanlan.zhihu.com/p/26639135

https://blog.csdn.net/yzxnuaa/article/details/79723187

https://blog.csdn.net/liuxiao214/article/details/74502975

https://blog.csdn.net/u011534057/article/details/54845673

https://www.leiphone.com/news/201701/yZvIqK8VbxoYejLl.html?viewType=weixin

https://www.sohu.com/a/161570126_775742


#### **框架**

tensorflow（python3.6）

tensorflow-gpu（python3.6）


#### **开源项目**

https://github.com/carpedm20/DCGAN-tensorflow.git

利用DCGAN进行图片自动补全：
https://github.com/saikatbsk/ImageCompletion-DCGAN.git

#### **训练数据集**

MNIST：http://yann.lecun.com/exdb/mnist/

CELEBA 人脸数据集：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

LSUN 场景数据集：https://www.yf.io/p/lsun

##### [GPU环境配置.md](ZXY_Example/gpu环境配置.md)

https://blog.csdn.net/qq_30611601/article/details/79067982


#### 





