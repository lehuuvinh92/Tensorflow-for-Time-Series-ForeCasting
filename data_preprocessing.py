# -*- coding: utf-8 -*-
# Created on Fri Jan 12 18:39:59 2018
# @author: acer
# =====================================

"""Preprocessing dataset module."""

import pandas as pd

"""Read dataset function"""
def read_dataset(dataset_path):
	# Read dataset from csv file
	data = pd.read_csv(dataset_path)
	# Drop data variable
	data = data.drop(['Date'], 1)
	# Make data a numpy array
	data = data.values
	return data