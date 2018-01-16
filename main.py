# -*- coding: utf-8 -*-
# Created on Fri Jan 12 18:37:45 2018
# @author: acer
# =====================================

"""Main module."""

#import preprocessing_dataset as pd

#tsf = pd.read_dataset("cpiai.csv")


import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

random.seed(111)
rng = pd.date_range(start='2000', end='2100', period=209, freq='M')
ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
ts.plot(c='b', title= "Example Time Series")
plt.show()
ts.head(10)
