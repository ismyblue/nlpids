# Name: balancer_gan.py
# Author: HuangHao
# Time: 2021/3/6 20:30

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Embedding
from preprocessor import PreProcessor
from keywords import get_keywords_dict, keywords_dict_size
from dataset import load_http_dataset_csic_2010
from sklearn.model_selection import train_test_split

"""
一个样本平衡器，可以通过此py文件获取一个样本平衡器，平衡样本
"""


class Balancer:
    """
    一个样本平衡器，调用generate_data生成指定数据
    """

    def __init__(self, data_size, vocab_size, label_number, model_filepath=None):
        """
        构造函数，构造一个样本平衡器
        :param data_size: 样本向量大小
        :param vocab_size: 样本字典大小
        :param label_number: 条件个数
        :param model_filepath: 平衡器模型路径
        """
        self.vocab_size = vocab_size
        self.data_size = data_size
        self.label_number = label_number
        self.model_filepath = model_filepath

        # 构造一个条件生成器
        self.generator = self.build_generator()
        # 构造一个条件判别器
        self.discriminator = self.build_discriminator()
        # self.generator.summary()
        # self.discriminator.summary()

        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                                   loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])

        self.discriminator.trainable = False
        # 噪声(None, data_size)
        noise = Input(shape=(self.data_size,), name='noise')
        # condition_label = Input(shape=(1,), dtype='int32', name='condition_label')
        # 生成合成样本
        # fake_data = self.generator([noise, condition_label])
        fake_data = self.generator(noise)
        # 判断真假
        # validity = self.discriminator([fake_data, condition_label])
        validity = self.discriminator(fake_data)
        # 固定住discriminator的http-CGAN, 输入输出，
        # self.http_cgan = Model([noise, condition_label], validity, name='http_cgan')
        self.http_cgan = Model(noise, validity, name='http_cgan')
        self.http_cgan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                               loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
        # self.http_cgan.summary()

    def build_generator(self):
        """
        构造一个条件生成器，生成unifited_format http digital数据
        :return:
        """
        # 构建一个序列模型
        model = Sequential()
        # 词嵌入层，word2vec
        # model.add(Embedding(self.vocab_size, 16, mask_zero=True))
        # 双向LSTM层
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)))
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)))
        # 全连接层
        model.add(Dense(self.data_size, activation='relu'))
        model.add(Dense(self.data_size, activation='relu'))
        model.add(Dense(self.data_size, activation='relu'))
        model.add(Dense(self.data_size, activation='relu'))
        model.add(Dense(self.data_size, activation='relu'))
        model.add(Dense(self.data_size, activation='relu'))
        # 输出
        model.add(Dense(self.data_size, activation='sigmoid'))

        # 输入噪声和标签[noise, label],接下来将两者合成(None, self.data_size)
        # 噪声大小self.data_size
        noise = Input(shape=(self.data_size,))
        # label = Input(shape=(1,), dtype='int32')

        # label (None, 1)->onehot编码后(None, 1, self.label_number)与Embedding层:(None, self.label_number, self.data_size)相乘，
        # 转换维度为(None, 1, self.data_size), 经过Flatten层:(None, 1, self.data_size)->(None, self.data_size)
        # label_vector = Flatten()(Embedding(self.label_number, self.data_size)(label))

        # noise与label合二为一，model_input (None, self.data_size)
        # model_input = tf.keras.layers.multiply([noise, label_vector])
        # 输入noise和label，获得合成图片fake_image
        # fake_data = model(model_input)
        fake_data = model(noise)
        # 构建判别器，输入噪声和标签:[(None, self.data_size), (None, 1)] 输出合成样本fake_data (None, self.data_size)
        # return Model([noise, label], fake_data, name='Genetator')
        return Model(noise, fake_data, name='Genetator')

    def build_discriminator(self):
        """
        构造一个条件判别器
        :return:
        """

        # 构建一个序列模型
        model = Sequential()
        # 词嵌入层，word2vec
        # model.add(Embedding(self.vocab_size, 16, mask_zero=True))
        # 双向LSTM层
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)))
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)))
        # 全连接层
        model.add(Dense(self.data_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        # 输出
        model.add(Dense(1, activation='sigmoid'))

        # 输入样本和标签[data, label]，接下来将两者合成 (None, self.data_size)
        data = Input(shape=(self.data_size,))  # (None, self.data_size)
        # label = Input(shape=(1,), dtype='int32')  # (None, 1)

        # label (None, 1)->onehot编码后(None, 1, self.label_number)与embedding(self.label_number, self.data_size)相乘，
        # 转换维度为(None, 1, self.data_size), 经过Flatten层(None, 1, self.data_size)-> label_vector (None, self.data_size)
        # label_vector = Flatten()(Embedding(self.label_number, self.data_size)(label))

        # data, model_input (None, self.data_size)
        # model_input = tf.keras.layers.multiply([data, label_vector])

        # 获取判别器输出结果 validity(正确性)
        # validity = model(model_input)
        validity = model(data)
        # 构建判别器,输入样本和标签:[(None, self.data_size), (None, 1)]  输出正确性validity (None, 1)
        # return Model([data, label], validity, name='Discriminator')
        return Model(data, validity, name='Discriminator')

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

        # pickle.dump(self.X_train, open('xtrain.pickle', 'wb'))
        # pickle.dump(self.Y_train, open('ytrain.pickle', 'wb'))

        # self.X_train = self.X_train / self.X_train.shape[1] / 2. - 1.

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

            flag = 0
            for real_data, real_labels in train_ds:
                # ############### 训练判别器discriminator #############################
                # 使用真实数据训练：x_train: [real_data, real_labels] [(batch_size, self.data_size), (batch_size, 1)],  y_train: real_validity (batch_size, 1)
                # metrics_real = self.discriminator.train_on_batch([real_data, real_labels], real_validity)

                metrics_real = self.discriminator.train_on_batch(real_data, real_validity)
                # metrics_real = self.discriminator.train_on_batch(real_data, real_labels)
                # print(metrics_real)

                # 使用合成数据训练
                # 噪声：noises (batch_size, self.data_size)
                noises = np.random.normal(0, 1, size=(batch_size, self.data_size))
                # 输入噪声和标签，利用生成器生成合成数据fake_data (None, self.data_size)
                # fake_data = self.generator.predict([noises, real_labels])
                fake_data = self.generator.predict(noises)
                # 转换为句向量
                for i in range(fake_data.shape[0]):
                    fake_data[i, :] = (fake_data[i, :] + 1)/2. * self.vocab_size
                fake_data = fake_data.astype('int32')

                # 使用合成数据训练：x_train: [fake_data, real_labels] [(batch_size, self.data_size), (batch_size, 1)],  y_train: fake_validity (batch_size, 1)
                # metrics_fake = self.discriminator.train_on_batch([fake_data, real_labels], fake_validity)
                metrics_fake = self.discriminator.train_on_batch(fake_data, fake_validity)

                # metrics_disc: loss 和 acc
                metrics_disc = 0.5 * np.add(metrics_real, metrics_fake)

                # ################ 训练生成器generator ###############################
                # 噪声: noises (batch_size, self.data_size)
                noises = np.random.normal(0, 1, size=(batch_size, self.data_size))
                # 标签：labels (batch_size, 1)
                # labels = np.random.randint(0, self.label_number, size=(batch_size, 1))
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
        # 条件标签：指定生成4个异常样本0，4个正常样本1
        # condition_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)

        # 生成器输入噪声和条件标签，输出合成样本
        # fake_data = self.generator.predict([noises, label])
        fake_data = self.generator.predict(noises)

        # 展示生成的样本
        pre = PreProcessor(get_keywords_dict())
        print('展示生成器生成的样本:')
        for i in range(fake_data.shape[0]):
            fake_data[i, :] = (fake_data[i, :] + 1)/2. * self.vocab_size
            fake_data[i, :] = fake_data[i, :].astype('int32') - 1
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


if __name__ == '__main__':
    data_balancer = Balancer(523, keywords_dict_size(), 2)
    (data, labels), (_, _) = load_http_dataset_csic_2010()
    data_balancer.train(data, labels, test_size=0.1, batch_size=32, epochs=20)
