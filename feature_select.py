# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 06:49:37 2020

@author: k5000751
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
##Filter methods
#Function to take a df with num attributes and cat target and return df with k_out_features
def feat_sel_Num_to_Cat(X, y, k_out_features):
    fs=SelectKBest(score_func=f_classif, k=k_out_features)
    df_sel=fs.fit_transform(X, y)
    if k_out_features=='all':
        for i in range(len(fs.scores_)):
            print('Feature of feat_sel_Num_to_Cat %s: %f' % (X.columns[i], fs.scores_[i]))
    #we have to create a dataframe
    return df_sel

#Function to take a df with cat attributes and cat target and return df with k_out_features
def feat_sel_Cat_to_Cat(X, y, k_out_features):
    #chi-squared feature selection
    fs_chi2=SelectKBest(score_func=chi2, k=k_out_features)
    df_chi2=fs_chi2.fit_transform(X,y)
    if k_out_features=='all':
        for i in range(len(fs_chi2.scores_)):
            print('Feature of feat_sel_Cat_to_Cat chi2 %s: %f' % (X.columns[i], fs_chi2.scores_[i]))
    
    #Mutual information feature selection
    fs_mutinf=SelectKBest(score_func=mutual_info_classif, k=k_out_features)
    df_mutinf=fs_mutinf.fit_transform(X,y)
    if k_out_features=='all':
        for i in range(len(fs_mutinf.scores_)):
            print('Feature of feat_sel_Cat_to_Cat mutual info %s: %f' % (X.columns[i], fs_mutinf.scores_[i]))
   
    return df_chi2, df_mutinf

##Wrapper methods


##Embedded methods