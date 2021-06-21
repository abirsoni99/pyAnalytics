# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:44:35 2021

@author: asabi
"""
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt
plt.style.use('ggplot')


#readfile
mtcars=pd.read_csv('mtcars.csv')
mtcars
mtcars.columns
#Rename Column 0 to Name
Col1=mtcars.columns[0]
Col1
mtcars = mtcars.rename(columns = {Col1:"Name"})
#skewness
mtcars.skew(axis=0)
#kurtosis
mtcars.kurt()      
columns_here=list(['mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb'])
#histogram
plt.hist(mtcars.mpg,bin=15)
plt.hist(mtcars.cyl,bin=15)
plt.hist(mtcars.disp,bin=15)
plt.hist(mtcars.hp,bin=15)
plt.hist(mtcars.drat,bin=15)
plt.hist(mtcars.wt,bin=15)
plt.hist(mtcars.qsec,bin=15)
plt.hist(mtcars.vs,bin=15)
plt.hist(mtcars.am,bin=15)
plt.hist(mtcars.gear,bin=15)
plt.hist(mtcars.carb,bin=15)

#for i in range(1,12,1):
 #   plt.figure()
  #  #plt.title("Histogram for: ",mtcars.columns[i])
   # plt.hist(mtcars[mtcars.columns[i]],bin=15)

#Covariance
mtcars.cov()
#Correlation
mtcars.corr()

#Outlier Analysis
import seaborn as sns
#box plot
for i in range(1,12,1):
    print("Outliers for Columns: ",mtcars.columns[i]);
    plt.figure() #avoid superimposing
    sns.boxplot(x=mtcars[mtcars.columns[i]]).set(title='Boxplot for'+mtcars.columns[i])
    Q1 =mtcars[mtcars.columns[i]].quantile(0.25)
    Q3 = mtcars[mtcars.columns[i]].quantile(0.75)
    Q1, Q3
    IQR = Q3 - Q1
    IQR
    #outlier : < Q1 - IQR or > Q3 + IQR
    list1 = list((mtcars[mtcars.columns[i]] < Q1-1.5 * IQR) | (mtcars[mtcars.columns[i]] > Q3+ 1.5  *IQR))
    print(list1.count(True))
   
   
   