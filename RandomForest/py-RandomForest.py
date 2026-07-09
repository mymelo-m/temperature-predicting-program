from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

start_time = time.time()

#导入数据集
data = pd.read_excel('SH10yTemp.xlsx')
data2 = data.to_numpy()
data_train = data2[0:3287,0:5]
data_test = data2[3288:3653,0:5]
y_train = data2[0:3287,5]
y_test = data2[3288:3653,5]

# 评估回归性能
# criterion ：
# 回归树衡量分枝质量的指标，支持的标准有三种：
# 1）输入"mse"使用均方误差mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，
# 这种方法通过使用叶子节点的均值来最小化L2损失
# 2）输入“friedman_mse”使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差
# 3）输入"mae"使用绝对平均误差MAE（mean absolute error），这种指标使用叶节点的中值来最小化L1损失

#此处使用mse
forest = RandomForestRegressor(n_estimators=1000,
                               criterion='friedman_mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(data_train, y_train)
y_test_pred = forest.predict(data_test)

print('RMSE test: %f' % (
        np.sqrt(mean_squared_error(y_test, y_test_pred))))
print('R^2 test: %f' % (
        r2_score(y_test, y_test_pred)))

y_pred_true=pd.concat([pd.DataFrame(y_test_pred),pd.DataFrame(y_test)],axis = 1)#axis=1 表示按照列的方向进行操作，也就是对每一行进行操作。
y_pred_true.columns=['predictvalues', 'realvalues']
y_pred_true.to_excel(r'D:\肖牧瑶\SJTU\大一\Projet Scientifique\天气预测\MyProjects\RandomForest\预测值与真实值.xlsx',index = False)

data1 = y_pred_true
plt.subplots(figsize=(10,5))
plt.xlabel('天数', fontsize = 10)
plt.ylabel('气温', fontsize = 10)
plt.plot(data1.predictvalues, color = 'b', label = '预测值')
plt.plot(data1.realvalues, color = 'r', label = '真实值')
plt.legend(loc=0)
plt.savefig("squares.png",
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