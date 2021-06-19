# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score 
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X
