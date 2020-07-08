# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:16:20 2020

@author: k5000751
"""


#CKD_pipeline.py file aimed at reproduce performance of CKD_script through pipeline
#to improve modularity

## All necessary modules as well as different functions that will be used in this work are explicit here.
#import all neccesary modules
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

#import modules created 
import my_utils
import missing_val_imput
import feature_select
import preprocessing
import adhoc_transf

from sklearn.model_selection import train_test_split


#%matplotlib inline 


#importing file into a pandas dataframe# As being unable to extract data from it original source, the csv file is downloaded from
#https://www.kaggle.com/mansoordaku/ckdisease
path_data=r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease\Chronic_Kidney_Disease\kidney_disease.csv'
df=pd.read_csv(path_data)
df.head()
df.describe()
df['classification'].value_counts()

#Set column id as index
df.set_index('id', inplace=True)

# Lets see summary of data
df.describe()

#Looking at describe table we can see that there are some missing features that apparently have numerical values. Let's see the
#type of these features, apart from the proportion of non-null values
my_utils.info_adhoc(df)

#As seen above, there are some strange caracters in pcv feature, therefore we will explore every features' value to homogeneize it.
my_utils.df_values(df)

#############################
##Step 0 Train-Test splitting
#############################
#Before starting to clean data, lets split train set and data set

train_set,test_set=train_test_split(df, test_size=0.2, random_state=42)


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["classification"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
    
train_set['classification'].value_counts()
test_set['classification'].value_counts()
    

strat_train_set['classification'].value_counts()
strat_test_set['classification'].value_counts()

train_set_copy=train_set.copy()

X_train=train_set_copy.drop('classification',axis=1)
y_train=train_set_copy['classification'].copy()

#############################
##Step 1 Misspelling correction and Encoding target feature
#############################
#Correct any misspelling correction in y_train
def misspellingCorrector(df):
    df.iloc[:] = df.iloc[:].str.replace(r'\t','')
    df.iloc[:] = df.iloc[:].str.replace(r' ','')
    return df

y_train=misspellingCorrector(y_train)

label_enc=LabelEncoder()
y_train=label_enc.fit_transform(y_train)


#############################
##Step 2 Pipeline creation for data preparation
#############################

print('Creating the data preparation Pipeline')

# dataprep_pipe=Pipeline(steps=[('misspelling', adhoc_transf.misspellingTransformer('classification'),
#                               ('features_cast', FeatureUnion([(                                  
#                                   'num_cast',Numeric_Cast(numerical_features)),
#                                   ('cat_cast',Category_Cast(features_to_category))])),
#                               ('data_missing',()),
#                               ('feautures_encoding',()),
#                               ('features_selection',()),
#                               ('preprocessing',())])

numerical_features=['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
category_features= ['sg','al','su','rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
                               
pipeline_numeric_feat= Pipeline([('mispelling',adhoc_transf.misspellingTransformer()),
                                 ('features_cast',adhoc_transf.Numeric_Cast_Column()),
                                 ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median')),
                                 ('features_select',feature_select.Feature_Selector(y_train, strategy='filter_mutinf', k_out_features=5)),
                                 ('scaler', MinMaxScaler())
                        ])

pipeline_category_feat= Pipeline([('mispelling',adhoc_transf.misspellingTransformer()),
                                 ('features_cast',adhoc_transf.Category_Cast_Column()),
                                 ('data_missing',missing_val_imput.Category_Imputer(strategy='most_frequent')),
                                 ('encoding', OrdinalEncoder()),
                                 ('features_select',feature_select.Feature_Selector(y_train, strategy='filter_cat', k_out_features=6))
                        ])

dataprep_pipe=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_features),
                                 ('category_pipe',pipeline_category_feat, category_features)
                                ])

X_train1=pipeline_numeric_feat.fit_transform(X_train[numerical_features],y_train)
X_train1=pipeline_category_feat.fit_transform(X_train[category_features],y_train)

X_train1=dataprep_pipe.fit_transform(X_train,y_train)

#############################
##Step 3 Pipeline creation for model
#############################




