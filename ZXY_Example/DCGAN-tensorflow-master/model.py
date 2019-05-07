from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  #math.ceil(x)返回大于等于参数x的最小整数,即对浮点数向上取整
  return int(math.ceil(float(size) / float(stride))) 
class DCGAN(object):
  #初始化函数
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir='data'):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training. 
      y_dim: (optional) Dimension of dim for y. [None] #y维度
      z_dim: (optional) Dimension of dim for Z. [100] #z纬度
      # 生成器第一个卷积层 filters size
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      # 鉴别器第一个卷积层filters size
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      # 生成器全连接层units size
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      # 鉴别器全连接层units size
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      # image channel
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess #session
    self.crop = crop 

    self.batch_size = batch_size #批处理大小
    self.sample_num = sample_num #样本数量

    #输入输出的宽高
    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    #定义各种维度
    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : batch_norm()批数据归一化操作，有助于加快训练速度
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.data_dir = data_dir

    #判断数据集的名字是否是mnist，是的话则直接用load_mnist()函数加载数据，否则需要从本地data文件夹中读取数据，并将图像读取为灰度图
    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]
    else:
      # dir *.jpg
      #从本地文件读取数据
      self.data = glob(os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern))
      imreadImg = imread(self.data[0])
      if len(imreadImg.shape) >= 3: #判断是否是灰度图，若是彩色图
        self.c_dim = imread(self.data[0]).shape[-1] # 获取图片通道
      else:
        self.c_dim = 1 #若是灰度图c_dim=1

    self.grayscale = (self.c_dim == 1) # 是否是灰度图像，c_dim==1为灰度图

    self.build_model()

  def build_model(self):
    if self.y_dim:
      #tf.placeholder(),在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据
      #初始化y
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    #判断crop是否为真，是的话是进行测试，图像维度是输出图像的维度；否则是输入图像的维度
    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    #利用tf.placeholder定义inputs，是真实数据的向量
    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    #利用tf.placeholder定义生成器用到的噪音z，z_sum
    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    # 直方图可视化
    self.z_sum = histogram_summary("z", self.z)

    #用噪音z和标签y初始化生成器G、用输入inputs初始化判别器D和D_logits、样本、用G和y初始化D_和D_logits
    self.G                  = self.generator(self.z, self.y) #生成器
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False) #判别器
    self.sampler            = self.sampler(self.z, self.y) 
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)
    
    #sigmoid交叉熵损失函数
    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        #函数原型tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None,,labels=None,logits=None,name=None)
        #对于给定的logits计算sigmoid的交叉熵
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    #d_loss_real真实数据的判别损失值
    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    #d_loss_fake虚假数据的判别损失值
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    #g_loss生成器损失值
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    # scalar_summary:Outputs a `Summary` protocol buffer containing a single scalar value
    # 返回一个scalar
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    #d_loss判别器损失值
    self.d_loss = self.d_loss_real + self.d_loss_fake

    #调用ops.py中的scalar_summary,分别汇总和记录g_loss和d_loss的标量数据
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables() #返回需要训练的变量列表

    self.d_vars = [var for var in t_vars if 'd_' in var.name] # 鉴别器相关变量参数集
    self.g_vars = [var for var in t_vars if 'g_' in var.name] # 生成器相关变量参数集

    self.saver = tf.train.Saver() #保存模型

  def train(self, config):
    #定义判别器优化器d_optim和生成器优化器g_optim
    #tf.train.AdamOptimizer()此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    
    #分别将关于生成器和判别器有关的变量各合并到一个变量中，并写入事件文件中
    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    #噪音z的初始化
    #np.random.uniform()从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high
    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    

    #根据数据集是否为mnist的判断，进行输入数据和标签的获取。这里使用到了utils.py文件中的get_image函数
    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]
    else:
      # self.data is like:["0.jpg","1.jpg",...]
      sample_files = self.data[0:self.sample_num]
      sample = [
          # get_image返回的是取值为(-1,1)的,shape为(resize_height,resize_width)的ndarray
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        # 灰度图像的channel（通道）为1
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None] #变换数组类型
      else:
        # 彩色图片
        sample_inputs = np.array(sample).astype(np.float32) #变换数组类型
  
    #定义计数器counter和起始时间start_time
    counter = 1
    start_time = time.time()
    #加载检查点，并判断加载是否成功
    could_load, checkpoint_counter = self.load(self.checkpoint_dir) 
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

   
    #开始for epoch in xrange(config.epoch)循环训练。先判断数据集是否是mnist，获取批处理的大小
    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:
        # self.data is like:["0.jpg","1.jpg",...]
        self.data = glob(os.path.join(
          config.data_dir, config.dataset, self.input_fname_pattern))
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size
      
      #开始for idx in xrange(0, batch_idxs)循环训练，判断数据集是否是mnist，来定义初始化批处理图像和标签
      for idx in xrange(0, batch_idxs):
        if config.dataset == 'mnist':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            # add a channel for grayscale
            # batch_images shape:(batch,height,width,channel)
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None] #变换数组类型
          else:
            batch_images = np.array(batch).astype(np.float32)#变换数组类型
        # add noise
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        if config.dataset == 'mnist':
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
              self.y:batch_labels,
            })
          # 用于可视化
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # 运行生成器优化器两次，以确保判别器损失值不会变为0，然后是判别器真实数据损失值和虚假数据损失值、生成器损失值
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z, self.y:batch_labels })
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z, 
              self.y:batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels
          })
        else:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        #输出本次批处理中训练参数的情况，首先是第几个epoch，第几个batch，训练时间，判别器损失值，生成器损失值
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, config.epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        fp = open("loss.txt", "a+")
        fp.write("d_loss:"+errD_fake+errD_real+", g_loss:"+errG+"\n")
        fp.close()

        # np.mod:Return element-wise remainder of division.
        # 每100次batch训练后，根据数据集是否是mnist的不同，获取样本、判别器损失值、生成器损失值，
        # 调用utils.py文件的save_images函数，保存训练后的样本，并以epoch、batch的次数命名文件。
        # 然后打印判别器损失值和生成器损失值
        if np.mod(counter, config.print_every) == 1:
          if config.dataset == 'mnist':
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
              }
            )
            # 保存生成的样本
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          else:
            try:
              samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                },
              )
              save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
              # 打印判别器损失值和生成器损失值
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            except:
              print("one pic error!...")
        # 每500次保存一下checkpoint
        if np.mod(counter, config.checkpoint_every) == 2: # save checkpoint file
          self.save(config.checkpoint_dir, counter)

  #判别器函数discriminator
  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope: #在一个作用域 scope 内共享变量
      if reuse:
        scope.reuse_variables() #对scope利用reuse_variables()进行重利用

      #如果self.y_dim为假，则直接设置5层，前4层为使用lrelu激活函数的卷积层，最后一层是使用线性层
      if not self.y_dim:
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
        #最后返回h4和sigmoid处理后的h4
        return tf.nn.sigmoid(h4), h4

      #如果self.y_dim为真
      else:
        #将Y_dim变为yb，然后利用ops.py文件中的conv_cond_concat函数，连接image与yb得到x
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)
        
        #然后设置4层网络，前3层是使用lrelu激励函数的卷积层，最后一层是线性层
        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)
        
        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')
        #最后返回h3和sigmoid处理后的h3
        return tf.nn.sigmoid(h3), h3

  #生成器函数generator
  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope: #在一个作用域 scope 内共享变量

      #如果self.y_dim为假
      if not self.y_dim:
        #获取输出的宽和高，然后根据这一值得到更多不同大小的高和宽的对
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2) # 2 is stride
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # 获取h0层的噪音z，权值w，偏置值b
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        #利用relu激励函数。h1层，首先对h0层解卷积得到本层的权值和偏置值
        # 然后利用relu激励函数。h2、h3等同于h1。h4层，解卷积h3
        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
        #返回使用tanh激励函数后的h4
        return tf.nn.tanh(h4)

      ##如果self.y_dim为真
      else:
        #获取输出的高和宽，根据这一值得到更多不同大小的高和宽的对
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # 获取yb和噪音z
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        #h0层，使用relu激励函数，并与1连接。
        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)

        #h1层，对线性全连接后使用relu激励函数，并与yb连接。
        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        #h2层，对解卷积后使用relu激励函数，并与yb连接
        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        #返回解卷积、sigmoid处理后的h2
        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def sampler(self, z, y=None): # 采样测试
    with tf.variable_scope("generator") as scope: #在一个作用域 scope 内共享变量
      scope.reuse_variables() #对scope利用reuse_variables()进行重利用

      #若self.y_dim为假，则基本同生成器
      if not self.y_dim: # generator
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)

      #若self.y_dim为真，则基本同判别器
      else: # discriminator
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(
            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  #加载mnist数据集
  def load_mnist(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    return X/255.,y_vec

  @property # 可以当属性来用
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  #save函数，用于保存训练好的模型
  def save(self, checkpoint_dir, step):
    # save checkpoint files
    model_name = "DCGAN.model"
    #创建检查点文件夹，如果路径不存在，则创建
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    
    #保存在这个文件夹下
    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  #load函数，读取检查点文件
  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    #读取检查点，获取路径
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    #成功读取
    if ckpt and ckpt.model_checkpoint_path:
      # basename:Returns the final component of a pathname
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      #重新存储检查点
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      #计数
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0)) 
      #打印成功信息
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    
    #读取失败，没有路径，打印失败信息
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0