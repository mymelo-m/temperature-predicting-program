import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kendalltau
from scipy.stats import chi2_contingency

'''
data = pd.read_excel('SH10yTemp-5.xlsx')

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
data_scaled=pd.DataFrame(data_scaled)
data_scaled.to_excel(r'D:\肖牧瑶\SJTU\大一\Project Scientifique\天气预测\MyProjects\归一化.xlsx')
'''

data_scaled = pd.read_excel('Normalization.xlsx')
'''
#Kendall's tau-b
dat1 = data_scaled.x7
dat2 = data_scaled.y1

c = 0
d = 0
t_x = 0
t_y = 0
for i in range(len(dat1)):
    for j in range(i+1,len(dat1)):
        if (dat1[i]-dat1[j])*(dat2[i]-dat2[j])>0:
            c = c + 1
        elif (dat1[i]-dat1[j])*(dat2[i]-dat2[j])<0:
            d = d + 1
        else:
            if (dat1[i]-dat1[j])==0 and (dat2[i]-dat2[j])!=0:
                t_x = t_x + 1
            elif (dat1[i]-dat1[j])!=0 and (dat2[i]-dat2[j])==0:
                t_y = t_y + 1
                
tau_b = (c - d) / np.sqrt((c+d+t_x)*(c+d+t_y))
            
print('tau_b = {0}'.format(tau_b))            
print('kendalltau(dat1,dat2) =  {0}'.format(kendalltau(dat1,dat2)))
'''

data1 = pd.read_excel('天气.xlsx')
kt=chi2_contingency(data1)

print('卡方值=%.4f, p值=%.4f, 自由度=%i expected_freq=%s'%kt)

