# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:44:17 2020

@author: k5000751
"""
#This class must include as much as function needed depeding on the 
#mispelling or wrong character detected in the dataset
import pandas as pd
import numpy as np

def mispelling (df):
#Some fetures content seems to have the character \t.
#Let's remove such character for the sake of consistency
    for i in range(0, len(df.columns)):
        if df.dtypes[i]==np.object:
            df.iloc[:,i] = df.iloc[:,i].str.replace(r'\t','',inplace=True)
            df.iloc[:,i] = df.iloc[:,i].str.replace(r' ','')
    return df

def num_feat_cast(df, num_cat):
    #Lets convert pcv,wc and rc dtype to float64 dtype and if any strange character appears it turns to NAN
    # df['pcv']=pd.to_numeric(df['pcv'],errors='coerce')
    # df['wc']=pd.to_numeric(df['wc'],errors='coerce')   
    # df['rc']=pd.to_numeric(df['rc'],errors='coerce')
    for i in range(len(num_cat)):
        #df[i]=pd.to_numeric(df[i],errors='coerce')
        df.loc[:,num_cat[i]]=pd.to_numeric(df.loc[:,num_cat[i]],errors='coerce')
    return df

def cat_feat_cast(df, cat_features):
    #Lets convert rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane to category
    #also features sg, al, su will be set to category
    for i in range(len(cat_features)):
        df.loc[:,cat_features[i]]=df.loc[:,cat_features[i]].astype('category')
    #df.info() 
    return df 

def target_to_cat(df, target):
    df.loc[:,target]=df.loc[:,target].astype('category')
    return df