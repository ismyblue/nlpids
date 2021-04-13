"""
重采样技术：过采样、欠采样、两者结合、...
"""

import pickle
from dataset import *
from keywords import keywords_dict_size
from classifier_multihead import WebAttackClassifier, CustomSchedule

from tensorflow import keras
from tensorflow.keras.metrics import Metric
from sklearn.utils import shuffle

from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

from callback_utils import Confusion_Matrix_Saver


def balance_data(method, x_train, y_train):
    """
    平衡样本
    :param method:
    :param x_train:
    :param y_train:
    :return: x_train, y_train
    """
    """    
    # 过采样：RandomOverSampler/SMOTE/ADASYN/BorderlineSMOTE
    # 欠采样：RandomUnderSampler/ClusterCentroids/NearMiss/TomekLinks
    # 两者结合：SMOTETomek/SMOTEENN
    """
    if method == 'None':
        return x_train, y_train
    elif method == 'RandomOverSampler':
        resamper = RandomOverSampler(random_state=0)
        x_train, y_train = resamper.fit_resample(x_train, y_train)
    elif method == 'SMOTE':
        resamper = SMOTE(random_state=0)
        x_train, y_train = resamper.fit_resample(x_train, y_train)
    elif method == 'ADASYN':
        resamper = ADASYN(random_state=0)
        x_train, y_train = resamper.fit_resample(x_train, y_train)
    elif method == 'BorderlineSMOTE':
        resamper = BorderlineSMOTE(random_state=0)
        x_train, y_train = resamper.fit_resample(x_train, y_train)
    elif method == 'RandomUnderSampler':
        resamper = RandomUnderSampler(random_state=0)
        x_train, y_train = resamper.fit_resample(x_train, y_train)
    elif method == 'ClusterCentroids':
        resamper = ClusterCentroids(random_state=0)
        x_train, y_train = resamper.fit_resample(x_train, y_train)
    elif method == 'NearMiss':
        resamper = NearMiss(version=2)
        x_train, y_train = resamper.fit_resample(x_train, y_train)
    elif method == 'TomekLinks':
        resamper = TomekLinks()
        x_train, y_train = resamper.fit_resample(x_train, y_train)
    elif method == 'SMOTETomek':
        resamper = SMOTETomek(random_state=0)
        x_train, y_train = resamper.fit_resample(x_train, y_train)
    elif method == 'SMOTEENN':
        resamper = SMOTEENN(random_state=0)
        x_train, y_train = resamper.fit_resample(x_train, y_train)
    # 洗牌，打乱
    x_train, y_train = shuffle(x_train, y_train, random_state=100)
    return x_train, y_train


if __name__ == '__main__':
    """
    样本平衡选择实验
    """
    # 获取字典大小
    vocab_size = keywords_dict_size()
    print("vocab_size:", vocab_size)

    resampling_methods = ['RandomOverSampler', 'RandomUnderSampler', 'SMOTE', 'SMOTEENN', 'None', 'TomekLinks',
                          'ADASYN', 'NearMiss', 'SMOTETomek', 'BorderlineSMOTE', 'ClusterCentroids']
    # 选用不同的方法进行样本平衡
    for i in range(0, len(resampling_methods)):
        method = resampling_methods[i]
        # 获取数据
        (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = load_http_dataset_csic_2010()
        x_train = x_train[:100]
        y_train = y_train[:100]
        x_validate = x_validate[:100]
        y_validate = y_validate[:100]
        x_test = x_test[:100]
        y_test = y_test[:100]
        print('样本平衡前: y_train', sorted(Counter(y_train).items()))

        if os.path.exists('resampling_data/{}.pkl'.format(method)):
            data = pickle.load(open('resampling_data/{}.pkl'.format(method), 'rb'))
            x_train, y_train = data['x_train'], data['y_train']
        else:
            # 样本平衡
            print('使用{}方法进行样本平衡'.format(method))
            x_train, y_train = balance_data(method, x_train, y_train)
            # 保存平衡后的样本
            print('保存{}方法平衡后的样本..'.format(method))
            with open('resampling_data/{}.pkl'.format(method), 'wb') as f:
                pickle.dump({'x_train': x_train, 'y_train': y_train}, f)
        print('{}算法样本平衡后：y_train'.format(method), sorted(Counter(y_train).items()))

        # one_hot 标签
        y_train = tf.squeeze(tf.one_hot(y_train, 2))
        y_validate = tf.squeeze(tf.one_hot(y_validate, 2))
        y_test = tf.squeeze(tf.one_hot(y_test, 2))

        # 确定随机种子,保证每个新的分类器初始化参数一致
        np.random.seed(100)
        tf.random.set_seed(100)

        # 构造或恢复分类器
        checkpoint_dir = "checkpoints/" + method
        # 构造分类器
        webAttackClassifier = WebAttackClassifier(num_layers=1, seq_len=x_train.shape[1], d_model=64, num_heads=8,
                                                  dff=16, input_vocab_size=vocab_size, categories=2)

        # # 学习速率
        # learning_rate = CustomSchedule(webAttackClassifier.d_model)
        # # 优化器
        # optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        optimizer = keras.optimizers.Adam()
        # 指定模型优化器、loss、指标
        webAttackClassifier.compile(optimizer=optimizer,
                                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                                    metrics=[tf.keras.metrics.CategoricalAccuracy(),
                                             tf.keras.metrics.Precision(),
                                             tf.keras.metrics.Recall(),
                                             ])

        # 训练##############################################
        print('开始训练...')
        # history记录回调函数
        csv_logger = tf.keras.callbacks.CSVLogger('history/{}_fit.csv'.format(method))

        # 保存训练集混淆矩阵的回调函数
        train_confusion_matrix_saver = Confusion_Matrix_Saver('confusion_matrix/{}_train.pkl'.format(method),
                                                              num_classes=2, x=x_train, y=y_train)
        # 保存验证集混淆矩阵的回调函数
        validate_confusion_matrix_saver = Confusion_Matrix_Saver('confusion_matrix/{}_validate.pkl'.format(method),
                                                                 num_classes=2, x=x_validate, y=y_validate)
        # 保存模型的回调函数
        model_saver = keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/" + method + "{epoch}",
                                                      save_freq='epoch', verbose=1)
        # 开始训练
        webAttackClassifier.fit(x_train, y_train, batch_size=64, epochs=5,
                                callbacks=[csv_logger, train_confusion_matrix_saver, validate_confusion_matrix_saver,
                                           model_saver])

        # 评估##############################################
        print('使用测试集合评估...')
        # 测试集的混淆矩阵保存
        test_confusion_matrix_saver = Confusion_Matrix_Saver('confusion_matrix/{}_test.pkl'.format(method),
                                                        num_classes=2, x=x_test, y=y_test)
        # 评估
        webAttackClassifier.evaluate(x_test[:10], y_test[:10], batch_size=64, callbacks=[test_confusion_matrix_saver])

        # 删除空间，避免OOM?
        del x_train
        del y_train
        del x_validate
        del y_validate
        del x_test
        del y_test
        del train_confusion_matrix_saver
        del validate_confusion_matrix_saver
        del test_confusion_matrix_saver
        del webAttackClassifier
