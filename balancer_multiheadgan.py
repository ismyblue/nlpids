# Name: balancer_gan.py
# Author: HuangHao
# Time: 2021/4/6 10:34

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Embedding
from tensorflow.keras.datasets import mnist
from classifier_multihead import MultiHeadAttention, MultiHeadLayer, MultiHeadClassifier


from preprocessor import PreProcessor
from keywords import get_keywords_dict, keywords_dict_size
from dataset import load_csic_2010
from sklearn.model_selection import train_test_split

"""
一个样本平衡器，可以通过此py文件获取一个样本平衡器，平衡样本
"""


class Balancer:
    """
    一个样本平衡器，调用generate_data生成指定数据
    """

    def __init__(self, data_size, d_model, vocab_size, num_classes, num_layers, num_heads, dff, rate=0.1, model_filepath=None):
        """
        构造函数，构造一个样本平衡器
        :param data_size: 样本向量大小
        :param d_model: 嵌入词向量大小
        :param vocab_size: 样本字典大小
        :param num_classes: 个数
        :param model_filepath: 平衡器模型路径
        """
        self.vocab_size = vocab_size
        self.data_size = data_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.model_filepath = model_filepath

        # 构造一个生成器
        self.generator = self.build_generator()
        # 构造一个判别器
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                                   loss="binary_crossentropy", metrics=['acc'])

        # 噪声(None, data_size)
        noise = Input(shape=(self.data_size,), name='noise')
        # # 条件
        # condition_label = Input(shape=(1,), dtype='int32', name='condition_label')
        # 生成合成样本
        # fake_data = self.generator([noise, condition_label])
        fake_data = self.generator(noise)
        # fake_data = tf.argmax(fake_data, axis=-1)

        # 判断真假
        # validity = self.discriminator([fake_data, condition_label])
        print('init', fake_data.shape)
        validity = self.discriminator(fake_data)
        # 固定住discriminator的http-CGAN, 输入输出
        self.discriminator.trainable = False
        # self.http_cgan = Model([noise, condition_label], validity, name='http_cgan')
        self.http_cgan = Model(noise, validity, name='http_cgan')

        self.http_cgan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                               loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])

        self.generator.summary()
        # self.discriminator.build(input_shape=(data_size,))
        # self.discriminator.summary()
        self.http_cgan.summary()

    def build_generator(self):
        """
        构造一个生成器，生成unifited_format http digital数据
        :return:
        """
        generator = Generator(self.data_size, self.num_layers, self.d_model, self.num_heads, self.dff, self.vocab_size)
        noise = Input(shape=(self.data_size,))  # (None, self.data_size)
        # condition_label = Input(shape=(1,), dtype='int32')
        # output = generator([noise, condition_label])
        output = generator(noise)

        return Model(noise, output, name='Genetator')

    def build_discriminator(self):
        """
        构造一个判别器,使用多头注意力编码器进行判别
        :return:
        """
        discriminator = Discriminator(self.num_layers, self.d_model, self.num_heads, self.dff, self.vocab_size, self.data_size)
        data = Input(shape=(self.data_size,))  # (None, self.data_size)

        output = discriminator(data)
        # return discriminator
        return Model(data, output, name='Discriminator')

    def train(self, data, labels, test_size, batch_size, epochs):
        """
        训练生成对抗网络
        :param data: 样本集
        :param labels: 标签
        :param test_size: 切分样本大小
        :param batch_size: 批次
        :param epochs: 迭代次数
        :return:
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(data, labels, test_size=test_size,
                                                                                random_state=21)

        real_validity = np.ones((batch_size, 1), dtype='int32')
        fake_validity = np.zeros((batch_size, 1), dtype='int32')

        batch_count = 0
        discriminator_losses = []
        generator_losses = []
        discriminator_accs = []
        generator_accs = []
        for epoch in range(epochs):
            # 数据集迭代器
            train_ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.Y_train)).shuffle(self.X_train.shape[0]) \
                .batch(batch_size, True)

            for real_data, real_labels in train_ds:
                # ############### 训练判别器discriminator #############################
                # 使用真实数据训练：x_train
                # metrics_real = self.discriminator.train_on_batch([real_data, real_labels], real_validity)
                metrics_real = self.discriminator.train_on_batch(real_data, real_validity)

                # 使用合成数据训练
                # 噪声：noises (batch_size, self.data_size)
                noises = np.random.normal(0, 1, size=(batch_size, self.data_size))
                # 输入噪声和标签，利用生成器生成合成数据fake_data (None, self.data_size)
                # fake_data = self.generator.predict([noises, real_labels])
                fake_data = self.generator.predict(noises)
                # fake_data = tf.argmax(input=fake_data, axis=-1)
                # metrics_fake = self.discriminator.train_on_batch([fake_data, real_labels], fake_validity)
                metrics_fake = self.discriminator.train_on_batch(fake_data, fake_validity)

                # metrics_disc: loss 和 acc
                metrics_disc = 0.5 * np.add(metrics_real, metrics_fake)
                # print(metrics_disc)

                # ################ 训练生成器generator ###############################
                # 噪声: noises (batch_size, self.data_size)
                noises = np.random.normal(0, 10, size=(batch_size, self.data_size))

                # 标签：labels (batch_size, 1)
                # labels = np.random.randint(0, self.num_classes, size=(batch_size, 1))
                # 使用噪声和标签训练生成器: x_train: [noises, label] [(batch_size, self.data_size), (batch_size, 1)], y_train: real_validity (batch_size, 1)
                # metrics_gen = self.http_cgan.train_on_batch([noises, labels], real_validity)
                metrics_gen = self.http_cgan.train_on_batch(noises, real_validity)

                print('batches:{} D_loss:{:.6f} D_acc:{:.2f}%, G_loss:{:.6f} G_acc:{:.2f}%'
                      .format(batch_count, metrics_disc[0], 100 * metrics_disc[1], metrics_gen[0],
                              100 * metrics_gen[1]))
                if batch_count % 10 == 0:
                    discriminator_losses.append(metrics_disc[0])
                    discriminator_accs.append(100 * metrics_disc[1])
                    generator_losses.append(metrics_gen[0])
                    generator_accs.append(100 * metrics_gen[1])

                # 测试generator效果,每一百批进行测试
                if batch_count % 100 == 0:
                    self.generate_data([0, 0, 1, 1])

                batch_count += 1

            # 结束一epoch保存权重
            if not os.path.exists('cweights'):
                os.mkdir('cweights')
            self.http_cgan.save('./cweights/Epoch{}_D_loss{:.6f}_D_acc{:.2f}_G_loss{:.6f}_G_acc{:.2f}.h5'
                                .format(epoch, metrics_disc[0], 100 * metrics_disc[1], metrics_gen[0],
                                        100 * metrics_gen[1]))
            # self.save_plt(discriminator_losses, generator_losses, discriminator_accs, generator_accs, epoch)

    def generate_data(self, label):
        """
        生成数据
        :param label:
        :return:
        """
        # 噪声
        noises = np.random.normal(0, 1, (len(label), self.data_size))
        label = np.array(label)
        # 标签：指定生成4个异常样本0，4个正常样本1
        # condition_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)

        # 生成器输入噪声和标签，输出合成样本
        # fake_data = self.generator.predict([noises, label])
        fake_data = self.generator.predict(noises)

        # 展示生成的样本
        pre = PreProcessor(get_keywords_dict())
        print('展示生成器生成的样本:')
        for i in range(fake_data.shape[0]):
            # fake_data = tf.argmax()
            fake_data = tf.argmax(fake_data, axis=-1)
            print(pre.digital_to_words(fake_data[i, :]))

    def save_plt(self, discriminator_losses, generator_losses, discriminator_accs, generator_accs, batch_count):
        # 画loss
        plt.figure()
        plt.plot(np.arange(0, len(discriminator_losses) * 10, 10), discriminator_losses)
        plt.plot(np.arange(0, len(generator_losses) * 10, 10), generator_losses)
        plt.xlabel('batch')
        plt.ylabel('loss')
        plt.legend(['Discriminator loss', 'Generator loss'])
        plt.savefig('./cgan_mnist/loss_batch_count{}.png'.format(batch_count))
        # 画acc/
        plt.figure()
        plt.plot(np.arange(0, len(discriminator_accs) * 10, 10), discriminator_accs)
        plt.plot(np.arange(0, len(generator_accs) * 10, 10), generator_accs)
        plt.xlabel('batch')
        plt.ylabel('Accuracy')
        plt.legend(['Discriminator Accuracy', 'Generator Accuracy'])
        plt.savefig('./cgan_mnist/acc_batch_count{}.png'.format(batch_count))
        plt.cla()


class Generator(keras.Model):
    def __init__(self, data_size, num_layers, d_model, num_heads, dff,
                 input_vocab_size, rate=0.1):
        """
        模型构造
        :param num_layers: 注意力编码层数量
        :param d_model: 词向量长度
        :param num_heads: 注意力头数
        :param dff:
        :param input_vocab_size: 输入词字典大小
        :param categories: 分类的数量
        :param rate: dropout失活比率
        """
        super(Generator, self).__init__()

        self.d_model = d_model
        self.input_vocab_size = input_vocab_size

        self.first_layer = keras.layers.Dense(data_size*d_model, activation='relu')
        # self.reshape = keras.layers.Reshape(target_shape=(data_size, self.d_model))
        # self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        # self.final_layer = keras.layers.Dense(input_vocab_size)
        # self.encoder = keras.layers.Dense(data_size*d_model, activation='relu')
        self.final_layer = keras.layers.Dense(data_size)

    def call(self, inp, training):
        inp = self.first_layer(inp)
        # inp = self.reshape(inp)
        # encoder输出
        # enc_output = self.encoder(inp, training)
        # 分类输出

        # final_output = self.final_layer(enc_output)
        final_output = self.final_layer(inp)


        return final_output


class Discriminator(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, data_size, rate=0.1):
        """
        模型构造
        :param num_layers: 注意力编码层数量
        :param d_model: 词向量长度
        :param num_heads: 注意力头数
        :param dff:
        :param input_vocab_size: 输入词字典大小
        :param categories: 分类的数量
        :param rate: dropout失活比率
        """
        super(Discriminator, self).__init__()

        self.d_model = d_model

        self.embedding = keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)
        # 词嵌入层，word2vec

        self.encoder = MultiHeadClassifier(num_layers, d_model, num_heads, dff, input_vocab_size, rate)

        self.reshape = keras.layers.Reshape(target_shape=(-1,))

        self.final_layer = keras.Sequential([
            keras.layers.Dense(523, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ], name="Final_layer")

    def call(self, inp, training):
        # word2vec
        # inp = self.embedding(inp)
        # # encoder输出
        # enc_output = self.encoder(inp, training)
        print(inp.shape)
        # enc_output = self.reshape(inp)
        # 分类输出
        final_output = self.final_layer(inp)

        return final_output



if __name__ == '__main__':
    data_balancer = Balancer(523, 16, keywords_dict_size(), num_classes=2, num_layers=1, num_heads=1, dff=16)
    (data, labels), (_, _) = load_csic_2010()
    data_balancer.train(data, labels, test_size=0.1, batch_size=64, epochs=20)
