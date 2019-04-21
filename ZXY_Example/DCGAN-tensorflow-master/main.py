import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

flags = tf.app.flags
# tensorflow 定义命令行参数，用于支持接受命令行传递参数，相当于接受argv，对各种类型的DEFINE函数中，第一个参数是参数名称，第二个参数是默认值，第三个是参数描述。

flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")  #迭代次数
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]") #学习速率，默认0.002
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]") #momentum用于梯度下降优化算法，指加速下降动量
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]") #图片大小
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]") #每次迭代的图像数量
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]") #ָ指定输入图像的高
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]") #指定输入图像的宽，没有则等于高
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]") #ָ指定输出图像的高
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]") #指定输出图像的宽，没有则等于高
flags.DEFINE_integer("print_every",100,"print train info every 100 iterations") #打印一次训练信息的迭代次数，默认100
flags.DEFINE_integer("checkpoint_every",500,"save checkpoint file every 500 iterations") #保存checkpoint文件的迭代次数，默认500
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]") #指定处理哪个数据集
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]") #输入图像文件格式
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]") #保存checkpoint的文件路径
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]") #数据集根目录
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]") #图像样本的保存路径
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]") #布尔值train，true表示在训练，false表示在测试
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]") #布尔值crop，true表示在训练，false表示在测试
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]") #布尔值visualize，true表示正在可视化
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]") #测试期间要生成的图像数
FLAGS = flags.FLAGS
#FLAGS是一个保存了解析后命令行参数的对象，获取参数只需例如：FLAGS.train_size 即可。

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  #print()采用分行打印输出，所以对于数据结构比较复杂、数据长度较长的数据，适合采用pprint()打印方式

  
  # 如果没有指定输入输出图像的宽，则等于高
  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height
  
  #判断checkpoint和sample的文件是否存在，不存在则创建
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto() #ConfigProto()用于配置tf.Session的运算方式
  run_config.gpu_options.allow_growth=True #当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存

  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist': #判断处理的是哪个数据集，然后对应使用不同参数的DCGAN类
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=10,
          z_dim=FLAGS.generate_test_images,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          z_dim=FLAGS.generate_test_images,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir)

    show_all_variables() #展示所有变量

    #判断是训练还是测试，如果是训练，则进行训练；如果不是，判断是否有训练好的model，然后进行测试，如果没有先训练，则会提示“[!] Train a model first, then run test mode”
    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
        # dcgan.load return:True,counter
      if not dcgan.load(FLAGS.checkpoint_dir)[0]: #没有checkpoint file，没有训练
        raise Exception("[!] Train a model first, then run test mode")


    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    #执行可视化方法，传入会话、DCGAN、配置参数，选项。
    OPTION = 1
    visualize(sess, dcgan, FLAGS, OPTION) 

if __name__ == '__main__':
  tf.app.run()