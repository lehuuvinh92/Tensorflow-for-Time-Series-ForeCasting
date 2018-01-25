# -*- coding: utf-8 -*-
# Created on Fri Jan 12 18:39:59 2018
# @author: acer
# =====================================

"""Preprocessing dataset module."""

import csv

"""Read dataset function"""
def read_dataset(dataset_path):
    Col1 = "Date"
    Col2 = "Index"
    Col3 = "Inflation"
    mydict = {Col1:[], Col2:[], Col3:[]}
    csvfile = csv.reader(open(dataset_path, "r"))
    for row in csvfile:
        mydict[Col1].append(row[0])
        mydict[Col2].append(row[1])
        mydict[Col3].append(row[2])
    return mydict

"""Define features function"""
def define_Features():
    print("he")
