import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

#定义一系列参数，先从tensorflow直接获取，获取失败则从tf.summary中获取
#tf.summay模块介绍https://blog.csdn.net/maweifei/article/details/80301020
try:
    image_summary = tf.image_summary #功能同下tf.summary.image
    scalar_summary = tf.scalar_summary #功能同下 tf.summary.scalar
    histogram_summary = tf.histogram_summary #功能同下tf.summary.histogram
    merge_summary = tf.merge_summary #功能同下tf.summary.merge
    SummaryWriter = tf.train.SummaryWriter #功能同下tf.summary.FileWriter

except:
    image_summary = tf.summary.image #函数原型image(name, tensor, max_outputs=3, collections=None, family=None)，输出一个包含图像的summary,这个图像是通过一个4维张量构建的，这个张量的四个维度如下所示：[batch_size,height, width, channels]
    scalar_summary = tf.summary.scalar #函数原型scalar(name, tensor, collections=None, family=None)输出一个含有标量值的Summary protocol buffer，这是一种能够被tensorboard模块解析的【结构化数据格式】
    histogram_summary = tf.summary.histogram #函数原型histogram(name, values, collections=None, family=None)，用来显示直方图信息，将【计算图】中的【数据的分布/数据直方图】写入TensorFlow中的【日志文件】
    merge_summary = tf.summary.merge #函数原型merge(inputs, collections=None, name=None)合并summaries，该op创建了一个summary协议缓冲区，它包含了输入的summaries的所有value的union
    SummaryWriter = tf.summary.FileWriter #函数原型FileWritter(path,sess.graph)，指定一个文件用来保存图。可以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中


#文件中判断concat_v2是否存在，存在则定义concat_v2()函数，否则定义concat()，用于连接多个矩阵。两者参数相同
#其中tensor:给定tensor数组；axis:合并的维度
#例
#t1 = [[1, 2, 3], [4, 5, 6]]
#t2 = [[7, 8, 9], [10, 11, 12]]
#tf.concat([t1, t2]，0) == > [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
#tf.concat([t1, t2]，1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

#定义batch_norm类
#Batchnorm是深度网络中经常用到的加速神经网络训练，加速收敛速度及稳定性的算法
#详细介绍：https://blog.csdn.net/qq_25737169/article/details/79048516，https://www.cnblogs.com/eilearn/p/9780696.html
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"): #初始化函数
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    # 定义了class 的__call__ 方法，可以把类像函数一样调用，利用tf.contrib.layers.batch_norm函数批处理规范化
    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)

#定义conv_cond_concat(x, y)，调用concat函数连接x,y与Int32型的[x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]]维度的张量乘积。
def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    # 沿axis = 3(最后一个维度连接)
    return concat([
        x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3) #调用concat函数

#conv2d卷积函数
def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        #tf.nn.conv2d是TensorFlow里面实现卷积的函数，参考资料http://www.cnblogs.com/qggg/p/6832342.html
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

# 反卷积函数deconv2d
def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):

    with tf.variable_scope(name): #指定卷积层作用域进行区分，名称为name
        # filter : [height, width, output_channels, in_channels]
        #tf.get_variable：若指定作用域/标识符的变量未创建，则创建新的变量；已创建则重用，这里标识符为'w'
        #函数原型tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)返回一个生成具有正态分布的张量的初始化器。
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            #做一个反卷积操作,tf.nn.conv2d_transpose是TensorFlow里面实现反卷积的函数，参考资料https://blog.csdn.net/mieleizhi0522/article/details/80441571
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            #另一种版本的反卷积函数
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        #tf.get_variable：若指定作用域/标识符的变量未创建，则创建新的变量；已创建则重用，这里标识符为'biases'
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        #函数原型tf.reshape(tensor,shape,name=None)，将tensor变换为参数shape形式，其中的shape为一个列表形式，参考资料：https://blog.csdn.net/m0_37592397/article/details/78695318
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        #判断with_w是否为真，真则返回解卷积、权值、偏置值，否则返回解卷积。
        if with_w:
            return deconv, w, biases
        else:
            return deconv

# leaky relu
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x) #返回x，leak*x中最大值

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    # 本质其实就是做了一个matmul....
    #get_shape()，只有tensor才可以使用这种方法，返回的是一个元组，需要通过as_list()的操作转换成list
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):#指定卷积层作用域进行区分

        #tf.get_variable：若指定作用域/标识符的变量未创建，则创建新的变量；已创建则重用，这里标识符为'Martrix'
        #调用tf.random_normal_initializer返回一个生成具有正态分布的张量的初始化器。
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))

        #tf.get_variable：若指定作用域/标识符的变量未创建，则创建新的变量；已创建则重用，这里标识符为'bias'
        #tf.constant_initializer()，对于一个scalar, list 或者 tuple或者一个 n维numpy array，将其全部元素都设定为相应的值，这里是bias_start
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))

        # tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
        # 作用是将矩阵a乘以矩阵b，生成a * b。
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias