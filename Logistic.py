import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.width', 1000)#加了这一行那表格的一行就不会分段出现了
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

#数据导入
df = pd.read_excel(io='...data.xlsx',sheet_name='data')
df.head()

#观察数据
sns.scatterplot(x='grade1',y='grade2',data=df,hue='label')
sns.pairplot(df,hue='label')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score, recall_score, f1_score, precision_score

#数据准备
X = df.drop('label',axis=1)
y = df['label']

#训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=50)

#数据预处理 归一化
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

#定义模型
log_model = LogisticRegression()

#训练模型
log_model.fit(scaled_X_train,y_train)

#预测数据
y_pred = log_model.predict(scaled_X_test)
print(y_pred)

#置信度评分
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

sigmoid_inputs = np.arange(-10,10)
sigmoid_outputs=sigmoid(sigmoid(sigmoid_inputs))
print("Sigmoid Function Input :: {}".format(sigmoid_inputs))
print("Sigmoid Function Output :: {}".format(sigmoid_outputs))

plt.plot(sigmoid_inputs,sigmoid_outputs)
plt.xlabel("Sigmoid Inputs")
plt.ylabel("Sigmoid Outputs")
plt.show()

#模型性能评估
print(classification_report(y_test, y_pred))
print('Accuracy score:', accuracy_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred, average='weighted'))
print('F1-score:', f1_score(y_test, y_pred, average='weighted'))
print('Precision score:', precision_score(y_test, y_pred, average='weighted'))
