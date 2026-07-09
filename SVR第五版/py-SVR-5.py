#使用数据：农历季节+三伏天/风力/风向/天气/季节

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import warnings

start_time = time.time()

data = pd.read_excel('SH10yTemp-5.xlsx')
data2 = data.to_numpy()
data_train = data2[0:3287,0:5]
data_test = data2[3288:3653,0:5]
y_train = data2[0:3287,5]
y_test = data2[3288:3653,5]

scaler = StandardScaler()
scaler.fit(data_train)
data_train_s=scaler.fit_transform(data_train)#对训练集中的特征变量进行标准化
data_test_s=scaler.fit_transform(data_test)#对测试集的特征变量进行标准化


model=SVR(kernel='rbf')#使用径向核（rbf）
model.fit(data_train_s,y_train)#模型估计
print(model.score(data_test_s,y_test))#计算拟合优度

model=SVR(kernel='poly',degree=2)#使用二次多项式核
model.fit(data_train_s,y_train)#模型估计
print(model.score(data_test_s,y_test))#计算拟合优度

model=SVR(kernel='poly',degree=3)#使用三次多项式核
model.fit(data_train_s,y_train)#模型估计
print(model.score(data_test_s,y_test))#计算拟合优度

model=SVR(kernel='sigmoid')#使用S型核
model.fit(data_train_s,y_train)#模型估计
print(model.score(data_test_s,y_test))#计算拟合优度

'''
param_grid = {'C':[0.01,0.1,1,10,50,100,150],'epsilon':[0.01,0.1,1,10],'gamma':[0.01,0.1,1,10]}#定义参数网格
kfold=KFold(n_splits=10,shuffle=True,random_state=1)#定义10折随机分组
model=GridSearchCV(SVR(),param_grid,cv=kfold)
model.fit(data_train_s,y_train)
print(model.best_params_)
'''

#model1 = model.best_estimator_ #结合最优超参数，重新定义最优model
model1 = SVR(kernel='rbf', gamma=0.1, C=150, epsilon=1)
model1.fit(data_train_s,y_train)
print("\n",model1.score(data_test_s,y_test),"\n") #计算测试集拟合优度

sigmoid_pred=model1.predict(data_test_s)#使用测试集预测气温

model1_rmse = np.sqrt(mean_squared_error(y_test,sigmoid_pred))    #RMSE
model1_mae = mean_absolute_error(y_test,sigmoid_pred)   #MAE
model1_r2 = r2_score(y_test, sigmoid_pred)  # R2
print("The RMSE of RBF_SVR: ", model1_rmse)
print("The MAE of RBF_SVR: ",model1_mae)
print("R^2 of RBF_SVR: ",model1_r2)

sigmoid_pred_true=pd.concat([pd.DataFrame(sigmoid_pred),pd.DataFrame(y_test)],axis = 1)#axis=1 表示按照列的方向进行操作，也就是对每一行进行操作。
sigmoid_pred_true.columns=['predictvalues', 'realvalues']
sigmoid_pred_true.to_excel(r'D:\肖牧瑶\SJTU\大一\Projet Scientifique\天气预测\MyProjects\SVR第五版\预测值与真实值.xlsx',index = False)

data1= sigmoid_pred_true

plt.subplots(figsize=(10,5))
plt.xlabel('天数', fontsize = 10)
plt.ylabel('气温', fontsize = 10)
plt.plot(data1.predictvalues, color = 'b', label = '预测值')
plt.plot(data1.realvalues, color = 'r', label = '真实值')
plt.legend(loc=0)
plt.savefig("squares-5.png",
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            dpi=300,
            orientation ='landscape')

end_time = time.time()
runtime = end_time-start_time
print("\nruntime = ",runtime)