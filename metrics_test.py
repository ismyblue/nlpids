# 指标测试
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score


def Evaluate1(y_test, y_predic):
    print('accuracy:', metrics.accuracy_score(y_test, y_predict))  # 预测准确率输出
    print('macro_precision:', metrics.precision_score(y_test, y_predict, average='macro'))  # 预测宏平均精确率输出
    print('micro_precision:', metrics.precision_score(y_test, y_predict, average='micro'))  # 预测微平均精确率输出
    # print('weighted_precision:', metrics.precision_score(y_test, y_predict, average='weighted')) #预测加权平均精确率输出
    print('macro_recall:', metrics.recall_score(y_test, y_predict, average='macro'))  # 预测宏平均召回率输出
    print('micro_recall:', metrics.recall_score(y_test, y_predict, average='micro'))  # 预测微平均召回率输出
    # print('weighted_recall:',metrics.recall_score(y_test,y_predict,average='weighted'))#预测加权平均召回率输出
    print('macro_f1:',
          metrics.f1_score(y_test, y_predict, labels=[0, 1, 2, 3, 4, 5, 6], average='macro'))  # 预测宏平均f1-score输出
    print('micro_f1:',
          metrics.f1_score(y_test, y_predict, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='micro'))  # 预测微平均f1-score输出
    # print('weighted_f1:',metrics.f1_score(y_test,y_predict,labels=[0,1,2,3,4,5,6],average='weighted'))#预测加权平均f1-score输出
    # target_names = ['class 1', 'class 2', 'class 3','class 4','class 5','class 6','class 7']
    # print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,y_predict,labels=[0,1,2,3,4,5,6]))#混淆矩阵输出 #比如[1,3]为2，即1类预测为3类的个数为2
    # print('分类报告:\n', metrics.classification_report(y_test, y_predict,labels=[0,1,2,3,4,5,6]))#分类报告输出 ,target_names=target_names


def Evaluate2(y_true, y_pred):
    print("accuracy:", accuracy_score(y_true, y_pred))  # Return the number of correctly classified samples
    print("macro_precision", precision_score(y_true, y_pred, average='macro'))
    print("micro_precision", precision_score(y_true, y_pred, average='micro'))
    # Calculate recall score
    print("macro_recall", recall_score(y_true, y_pred, average='macro'))
    print("micro_recall", recall_score(y_true, y_pred, average='micro'))
    # Calculate f1 score
    print("macro_f", f1_score(y_true, y_pred, average='macro'))
    print("micro_f", f1_score(y_true, y_pred, average='micro'))


y_test = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 0, 0, 0, 0]
y_predict = [1, 1, 1, 3, 3, 2, 2, 3, 3, 3, 4, 3, 4, 3, 5, 1, 3, 6, 6, 1, 1, 0, 6]
Evaluate1(y_test, y_predict)
Evaluate2(y_test, y_predict)

##其中列表左边的一列为分类的标签名，右边support列为每个标签的出现次数．avg / total行为各列的均值（support列为总和）．
##precision recall f1-score三列分别为各个类别的精确度/召回率及 F1值
'''
accuracy: 0.5217391304347826
macro_precision: 0.7023809523809524
micro_precision: 0.5217391304347826
macro_recall: 0.5261904761904762
micro_recall: 0.5217391304347826
macro_f1: 0.5441558441558441
micro_f1: 0.5217391304347826

accuracy: 0.5217391304347826
macro_precision 0.7023809523809524
micro_precision 0.5217391304347826
macro_recall 0.5261904761904762
micro_recall 0.5217391304347826
macro_f 0.5441558441558441
micro_f 0.5217391304347826
'''

