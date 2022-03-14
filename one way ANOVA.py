import statsmodels.api as sm
from  statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf
import statsmodels.stats.anova as sma

import numpy as np
import pandas as pd
import scipy.stats as st
import patsy
import seaborn as sns 
import matplotlib.pyplot as plt

A1  = np.array([10./47, 9./23, 8./21, 19./47, 17./41, 32./100, 16./50,26./68,27./67,8./34],dtype='float')
A2 = np.array([13./36, 18./54, 17./57,11./31, 12./37],dtype='float')
A3 = np.array([7./17, 37./103,20./61, 25./80, 22./78, 13./43, 22./63, 10./26],dtype='float')
A4  = np.array([16./43,12./30, 5./15, 7./20, 21./84, 49./143,36./115,15./42,16./50],dtype='float')
#数据总量
n=len(A1)+len(A2)+len(A3)+len(A4)
#水平数或组数
r=4
###各水平均值
mu1=np.mean(A1)
mu2=np.mean(A2)
mu3=np.mean(A3)
mu4=np.mean(A4)
#总均值
mu=(np.sum(A1)+np.sum(A2)+np.sum(A3)+np.sum(A4))/n
###组内平方和或误差平方和
se=np.sum((A1-mu1)**2)+np.sum((A2-mu2)**2)+np.sum((A3-mu3)**2)+np.sum((A4-mu4)**2)
###组间平方和或效应平方和
sa=len(A1)*(mu1-mu)**2+len(A2)*(mu2-mu)**2+len(A3)*(mu3-mu)**2+len(A4)*(mu4-mu)**2
###组内和组间均方，二者皆是实验数据方差的无偏估计。
mse=se/(n-r)
msa=sa/(r-1)
###F值
F=msa/mse
###F检验p值
pvalue=st.f.sf(F,r-1,n-r)

###打印各项统计量
print('                         单因素方差分析表')
print('方差来源    自由度  平方和      均方        F          P值')
print('因素A       %d       %6.6f    %6.6f    %6.6f   %6.6f'%(r-1,sa,msa,F,pvalue))
print('误差        %d      %6.6f    %6.6f    '%(n-r,se,mse))
print('总和        %d      %6.6f'%(n-1,se+sa))