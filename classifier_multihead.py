import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
from dataset import load_http_dataset_csic_2010
from keywords import keywords_dict_size

from classifier_BiLSTM import *


class Confusion_Matrix_Saver(keras.callbacks.Callback):
    """
    自定义回调函数，用于保存每epoch的混淆矩阵
    """
    def __init__(self, filepath, num_classes, x, y):
        """
        混淆矩阵保存回调
        :param filepath:
        :param num_classes:
        :param x: 数据
        :param y: 标签
        """
        self.filepath = filepath
        self.num_classes = num_classes
        self.x = x
        self.y = tf.argmax(y, axis=1)

    def on_train_begin(self, logs):
        self.confusion_matrix_list = []

    def on_epoch_end(self, epoch, logs):
        y_pre = self.model.predict(self.x)
        y_pre = tf.argmax(y_pre, axis=1)
        matrix = np.zeros(shape=(self.num_classes, self.num_classes), dtype='int32')
        for i in range(y_pre.shape[0]):
            matrix[self.y[i]][y_pre[i]] += 1
        self.confusion_matrix_list.append(matrix)
        # 每一epoch结束，保存一下所有的混淆矩阵
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.confusion_matrix_list, f)
        print('第{}个epoch的混淆矩阵保存成功...'.format(epoch))

    def on_test_begin(self, logs):
        self.confusion_matrix_list = []

    def on_test_end(self, logs):
        y_pre = self.model.predict(self.x)
        y_pre = tf.argmax(y_pre, axis=1)
        matrix = np.zeros(shape=(self.num_classes, self.num_classes), dtype='int32')
        for i in range(y_pre.shape[0]):
            matrix[self.y[i]][y_pre[i]] += 1
        self.confusion_matrix_list.append(matrix)
        # 每一epoch结束，保存一下所有的混淆矩阵
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.confusion_matrix_list, f)
        print('evaluate的混淆矩阵保存成功...')


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        """
        多头注意力
        :param d_model: 词向量长度
        :param num_heads: 多头数量
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        # 利用Dense层生成权重矩阵 wq wk wv
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)

        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v):
        """
        dot_product attention
        Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_kv, depth)
            v: value shape == (..., seq_len_kv, depth_v)
        :return: output shape == (..., seq_len_q, depth_v)
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # 转换数据类型, dk是k向量的维度
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logit = matmul_qk / tf.math.sqrt(dk)  # (..., seq_len_q, seq_len_kv)

        # (..., seq_len_q, seq_len_kv)
        attention_weights = tf.nn.softmax(scaled_attention_logit, axis=-1)
        # (..., seq_len_q, depth_v)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_q, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)

        # (batch_size, seq_len_q, d_model)
        scaled_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(scaled_attention)

        return output, attention_weights


class MultiHeadLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        编码层构造
        :param d_model: 词向量深度
        :param num_heads: 注意力头数量
        :param dff:
        :param rate: dropout失活比率
        """

        super(MultiHeadLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def point_wise_feed_forward_network(self, d_model, dff):
        """返回一个模型， Sequence是一个模型，有call方法"""
        return keras.Sequential([
            keras.layers.Dense(dff, activation='relu', kernel_initializer='he_uniform'),
            keras.layers.Dense(d_model)
        ])

    def call(self, x, training):
        # 传入x, x, x,分别利用x, x, x计算出q k v，再进行多头计算
        attn_output, attention_weights = self.mha(x, x, x)
        # multi-head 之后 dropout防止过拟合
        attn_output = self.dropout1(attn_output, training=training)
        # 残差连接，并layerNorm
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        # feed forward network 之后 dropout防止过拟合
        ffn_output = self.dropout2(ffn_output, training=training)
        # 残差连接，并layerNorm
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class MultiHeadClassifier(keras.layers.Layer):
    def __init__(self, num_layers, seq_len, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        """
        构造一个编码器
        :param num_layers: 编码层数量
        :param d_model: 词向量长度
        :param num_heads: 多头数量
        :param dff:
        :param input_vocab_size: 字典大小
        :param rate: dropout失活比率
        """
        super(MultiHeadClassifier, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = self.positional_encoding(seq_len + input_vocab_size, self.d_model)

        self.enc_layers = [MultiHeadLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

        # self.reshape = keras.layers.Reshape(target_shape=(-1, ))

    def get_angels(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angels(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # official version
        # pos_encoding = np.concatenate([sines, cosines], axis=-1)
        # pos_encoding = pos_encoding[np.newaxis, ...]

        # lsy version
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, training):
        """
        前向传播
        :param x:
        :param training: dropout是否训练状态
        :return:
        """

        # x = self.embedding(x)
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # sin cos位置
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)
        # 拉长
        # x = self.reshape(x)

        return x


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class WebAttackClassifier(keras.Model):
    def __init__(self, num_layers, seq_len, d_model, num_heads, dff,
                 input_vocab_size, categories, model_cls='Bi-LSTM_multihead', rate=0.1):
        """
        模型构造
        :param num_layers: 注意力编码层数量
        :param seq_len: 句向量长度
        :param d_model: 词向量长度
        :param num_heads: 注意力头数
        :param dff:
        :param input_vocab_size:  输入词字典大小
        :param categories: 分类的数量
        :param rate: dropout失活比率
        :param model_cls: 模型类别 'multihead' 'Bi-LSTM_multihead' 'Bi-LSTM'
        """
        super(WebAttackClassifier, self).__init__()
        self.model_cls = model_cls
        print('分类器模型架构为：{}'.format(model_cls))

        # 词嵌入层，word2vec
        self.embedding = keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)

        if 'multihead' in self.model_cls:
            self.multiheadclassifier = MultiHeadClassifier(num_layers, seq_len, d_model, num_heads, dff, input_vocab_size, rate)
            self.reshape = keras.layers.Reshape(target_shape=(-1,))

        if 'Bi-LSTM' in self.model_cls:
            self.rnn_layer = keras.Sequential([
                # 双向LSTM层
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
                # 全连接层
                keras.layers.Dense(16, activation='relu'),
            ], name='Bi-LSTM')

        self.final_layer = keras.Sequential([
            # keras.layers.Reshape(target_shape=(-1,)),
            keras.layers.Dense(8, activation='relu', kernel_initializer='he_uniform'),
            keras.layers.Dense(categories, activation='softmax')
        ], name="Final_layer")

    def call(self, inp, training):
        # word2vec
        inp = self.embedding(inp)
        # 模型类别  'multihead' 'Bi-LSTM_multihead' 'Bi-LSTM'

        # multiheadclassifier输出
        if 'multihead' in self.model_cls:
            enc_output = self.multiheadclassifier(inp, training)
            enc_output = self.reshape(enc_output)
        # Bi-LSTM输出
        if 'Bi-LSTM' in self.model_cls:
            rnn_output = self.rnn_layer(inp)

        # 合并输出
        if 'multihead' in self.model_cls and 'Bi-LSTM' in self.model_cls:
            output = tf.concat([enc_output, rnn_output], axis=1)
        elif 'multihead' in self.model_cls:
            output = enc_output
        elif 'Bi-LSTM' in self.model_cls:
            output = rnn_output

        # 分类输出
        final_output = self.final_layer(output)

        return final_output

    def train(self, x_train, y_train, x_test, y_test, epochs=20):
        """
        训练
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param epochs:
        :return:
        """
        # 学习速率
        learning_rate = CustomSchedule(self.d_model)
        # 优化器
        optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # 损失函数
        loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)#, reduction='none')
        # loss_object = keras.losses.BinaryCrossentropy(from_logits=True)#, reduction='none')

        train_loss = keras.metrics.Mean(name='train_loss')
        train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        # ************************************
        # 3. 训练流程的准备
        # ************************************

        # 定义checkpoints管理器
        checkpoint_path = "./checkpoints/train"
        ckpt = tf.train.Checkpoint(model=webAttackClassifier, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # 使用Tensorboard可视化训练过程的loss和accuracy
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # ************************************
        # 4. 开始训练
        # ************************************
        for epoch in range(epochs):
            start = time.time()
            train_loss.reset_states()
            train_accuracy.reset_states()

            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(y_train.shape[0]).batch(64)
            # 分批次训练
            for (batch, (inp, labels)) in enumerate(train_dataset):
                # 训练一批
                with tf.GradientTape() as tape:
                    predictions = webAttackClassifier(inp, training=True)
                    # loss = self.loss_function(labels, predictions, loss_object)
                    loss = tf.reduce_mean(loss_object(labels, predictions))

                gradients = tape.gradient(loss, webAttackClassifier.trainable_variables)
                optimizer.apply_gradients(zip(gradients, webAttackClassifier.trainable_variables))

                train_loss(loss)
                train_accuracy(labels, predictions)

                # 记录loss和accuracy
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

                if batch % 20 == 0:
                    print('Epochs {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            # 执行验证step
            val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(y_test.shape[0]).batch(64)
            self.validate(dataset=val_dataset)

            # 预测
            print("开始预测...")
            http_text_list = [http_text0, http_text1, http_text2, http_text3]
            pre = PreProcessor(get_keywords_dict())
            test_data = []
            for http_text in http_text_list:
                digital = pre.format_unification_digital_list(http_text)
                test_data.append(digital)
                print(len(digital), digital)
                print(pre.format_unification_text(http_text))
            # 填充定长
            test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=x_train.shape[1],
                                                                      padding='post')
            pre_labels = webAttackClassifier(test_data, training=False)
            print('pre_labels.shape:', pre_labels.shape)
            print(pre_labels)
            pre_labels = tf.argmax(tf.nn.softmax(pre_labels, axis=1), axis=1)
            print(pre_labels)
            for label in pre_labels:
                if label == 1:
                    print('异常', end=' ')
                else:
                    print('正常', end=' ')
            print()
            from tensorflow.keras.utils import plot_model

    def validate(self, dataset, eval_type='Val'):
        """ 验证或测试 """
        print("*** Running {} step ***".format(eval_type))
        accuracy = keras.metrics.SparseCategoricalAccuracy(name='{}_accuracy'.format(eval_type.lower()))

        for (batch, (inp, labels)) in enumerate(dataset):
            # tar_input = labels[:, :-1]  # 目标序列首位插入了开始符 [start]，相当于做了右移
            # tar_real = labels[:, 1:]  # 目标序列本身

            predictions = self.call(inp, training=False)

            accuracy(labels, predictions)

            # if batch % 100 == 0:
            if batch % 4 == 0:
                print('Batch {} Accuracy {:.4f}'.format(batch, accuracy.result()))
                break

        print('{} Accuracy {}\n'.format(eval_type, accuracy.result()))


from imblearn.combine import SMOTETomek
if __name__ == '__main__':

    model_cls_list = ['Bi-LSTM_multihead', 'multihead', 'Bi-LSTM']
    # 对不同的模型架构进行实验
    for model_cls in model_cls_list:
        print("开始对model_cls模型进行实验...")

        (x_train, y_train), (x_test, y_test) = load_http_dataset_csic_2010()
        print("x_train.shape:", x_train.shape)
        print("y_train.shape:", y_train.shape)
        vocab_size = keywords_dict_size()
        print("vocab_size:", vocab_size)
        # x_train = x_train[:100]
        # y_train = y_train[:100]
        # x_test = x_test[:100]
        # y_test = y_test[:100]

        # one_hot 标签
        y_train = tf.squeeze(tf.one_hot(y_train, 2))
        y_test = tf.squeeze(tf.one_hot(y_test, 2))

        # 确定随机种子
        np.random.seed(100)
        tf.random.set_seed(100)

        # 构造分类器
        webAttackClassifier = WebAttackClassifier(
            num_layers=1,
            seq_len=x_train.shape[1],
            d_model=16,
            num_heads=8,
            dff=16,
            input_vocab_size=vocab_size,
            categories=2,
            model_cls=model_cls
        )

        # 学习速率
        learning_rate = CustomSchedule(16)
        # 优化器
        optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # 指定模型优化器、loss、指标
        webAttackClassifier.compile(optimizer=optimizer,
                                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                                    metrics=[tf.keras.metrics.CategoricalAccuracy(),
                                             tf.keras.metrics.Precision(),
                                             tf.keras.metrics.Recall(),
                                             ])

        # 训练##############################################
        # history记录回调
        csv_logger = tf.keras.callbacks.CSVLogger('history/{}_fit.csv'.format(model_cls))
        # 保存混淆矩阵的回调
        confusion_matrix_saver = Confusion_Matrix_Saver('confusion_matrix/{}_fit.pkl'.format(model_cls), num_classes=2,
                                                        x=x_test, y=y_test)
        # 保存模型的回调
        model_saver = keras.callbacks.ModelCheckpoint(filepath="checkpoints/{}".format(model_cls),
                                                      save_freq='epoch', verbose=1)
        # 开始训练
        webAttackClassifier.fit(x_train, y_train, batch_size=64, epochs=50,
                                callbacks=[csv_logger, confusion_matrix_saver, model_saver])
        # webAttackClassifier.train(x_train, y_train, x_test, y_test, 20)

        # 评估##############################################
        print('开始评估...')
        # 混淆矩阵保存
        confusion_matrix_saver = Confusion_Matrix_Saver('confusion_matrix/{}_evaluate.pkl'.format(model_cls),
                                                        num_classes=2,
                                                        x=x_test, y=y_test)
        # 评估
        webAttackClassifier.evaluate(x_test, y_test, batch_size=64, callbacks=[confusion_matrix_saver])

        # 删除空间，避免OOM?
        del x_train
        del y_train
        del x_test
        del y_test
        del confusion_matrix_saver
        del webAttackClassifier