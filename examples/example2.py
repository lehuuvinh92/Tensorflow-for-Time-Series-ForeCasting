# -*- coding: utf-8 -*-
# Created on Sat Jan 20 14:51:00 2018
# @author: acer
# =====================================

"""Example 2 for time-series forecasting"""
import numpy as np

seq_len = 10

def create_time_series():
    freq = (np.random.random()*0.5) + 0.1 #0.1 to 0.6
    ampl = np.random.random() + 0.5 # 0.5 to 1.5
    x = np.sin(np.arange(0, seq_len) * freq) * ampl
    return x

