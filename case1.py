# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 19:41:26 2021

@author: asabi
"""
url= 'https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/denco.csv'
#dataurl
import pandas as pd
import numpy as np
#importing from a url
df = pd.read_csv(url)
df
#basic df functions
df.shape #rows and columns
df.columns #names
df.head(3)#first 3
df.describe()
len(df)
df.dtypes
#otheroperations
#change region to type category
df['region']=df['region'].astype('category')  
df.head(1)   
df.describe()
#categorycount
df.region.value_counts()
#plot category count
df.region.value_counts().plot(kind='bar')

#Problem 1
#most loyal customer
df.sort_values(['custname']) #sort
df.custname.value_counts() #frequency
df.custname.value_counts().sort_values(ascending=False) #frequency sort
df.custname.value_counts().sort_values(ascending=False).head(5) #top 5

#Problem 2
#max revenue customer
df.groupby('custname').revenue.sum().sort_values(ascending=False).head(5)
#max min purchase order and number of orders
df.groupby('custname').aggregate({'revenue': [np.sum,max,min,'count']})
#sorting ke liye
df.groupby('custname')['revenue'].aggregate([np.sum,max,min,'count']).sort_values(by='sum', ascending=False).head(5)

#Problem 3
#revenueperpartnum
df.columns
df.groupby('partnum')['revenue'].aggregate([np.sum]).sort_values(by="sum",ascending=False).head(5)
#top profit margin item
df.groupby('partnum')['margin'].aggregate([np.sum]).sort_values(by="sum",ascending=False).head(5)

#Problem 4
#Most Sold items
df.columns
df.groupby('partnum').size().sort_values(ascending=False).head(5)

#Problem 5
#max revenue region
df[['revenue','region']].groupby('region').sum().sort_values(by ='revenue',ascending =False).head(5).plot(kind='barh')
#learn use of options
