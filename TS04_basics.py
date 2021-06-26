#Topic Time Series Basic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%%
dates = pd.date_range('2020-09-01',periods=5, freq='D') #create dates from start date
help(pd.date_range)
pd.date_range('2020-09-01',periods=5, freq='2W') #fortnight
dates
sales = pd.Series([50,60,55,70,80], index=dates)
sales #series

#%%% Simple Moving Average (SMA)
#offset mean
ma3 = sales.rolling(window=3).mean()
ma3
ma3c = sales.rolling(window=3, center=True).mean()
(50+60+55)/3
(60+55+70)/3
ma3c
#%%% : shift 1 down and then find the daily diff
sales.shift(1)
sales - sales.shift(1)  #daily changes
#Arima for time series:
#pip install pmdarima --user  #see the syntax #restart session
#https://pypi.org/project/pmdarima/
from pmdarima.utils import c, diff
help(diff)
diff(sales, lag=1, differences=1) #diff from function
#lag- gap; difference is B-A (t1-t0)
sales - sales.shift(1)  #same
#%%%%%
diff(sales, lag=2, differences=1)
np.vstack((sales, sales.shift(2), sales - sales.shift(2)))
sales - sales.shift(2)
#lag is the gap : 1 with 3, 2 with 4 and so on
#%%%%%
sales2 = sales.copy()
diff(sales, lag=1, differences=1)
sales2 - sales2.shift(1)

diff(sales, lag=1, differences=2)
sales2 = sales2 - sales2.shift(1)
sales2 - sales2.shift(1)

diff(sales, lag=1, differences=2)


#end here