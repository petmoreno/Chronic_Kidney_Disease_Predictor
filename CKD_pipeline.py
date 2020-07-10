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
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
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

from sklearn.model_selection import cross_val_score

#Classifier models to use
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


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
test_set_copy=test_set.copy()

X_train=train_set_copy.drop('classification',axis=1)
y_train=train_set_copy['classification'].copy()

X_test=test_set_copy.drop('classification',axis=1)
y_test=test_set_copy['classification'].copy()

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
##Step 2 Feature Engineering
#############################
#Cross_val_score fails due to features al and su has only few samples of values 5.0. So we have to cast to previous category
#X_train.loc[:,'al'].replace(5,4,inplace=True)
#X_train.loc[:,'su'].replace(5,4,inplace=True)
#############################
##Step 3 Pipeline creation for data preparation
#############################

print('Creating the data preparation Pipeline')

numerical_features=['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
category_features= ['sg','al','su','rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
len(category_features)
pipeline_numeric_feat= Pipeline([('mispelling',adhoc_transf.misspellingTransformer()),
                                 ('features_cast',adhoc_transf.Numeric_Cast_Column()),
                                 ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median')),
                                 ('features_select',feature_select.Feature_Selector(strategy='wrapper_RFECV')),
                                 ('scaler', MinMaxScaler())
                        ])

pipeline_category_feat= Pipeline([('mispelling',adhoc_transf.misspellingTransformer()),
                                 ('features_cast',adhoc_transf.Category_Cast_Column()),
                                 ('data_missing',missing_val_imput.Category_Imputer(strategy='most_frequent')),
                                 ('cat_feat_engineering',adhoc_transf.CastDown()),
                                 ('encoding', OrdinalEncoder()),
                                 ('features_select',feature_select.Feature_Selector(strategy='wrapper_RFECV'))
                        ])

dataprep_pipe=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_features),
                                 ('category_pipe',pipeline_category_feat, category_features)
                                ])


#For testing data_prep pipelines individually
#X_train1=pipeline_numeric_feat.fit_transform(X_train[numerical_features],y_train)
#X_train1=pipeline_category_feat.fit_transform(X_train[category_features],y_train)

#X_train1=dataprep_pipe.fit_transform(X_train,y_train)

#############################
##Step 4 Pipeline creation for model
#############################
#Several classifier with Cross validation will be applied
y_test=misspellingCorrector(y_test)

label_enc=LabelEncoder()
y_test=label_enc.fit_transform(y_test)

sgd_clf=SGDClassifier()
logreg_clf=LogisticRegression()
linsvc_clf=LinearSVC()
svc_clf=SVC()
dectree_clf=DecisionTreeClassifier()
rndforest_clf=RandomForestClassifier()
#
print ('Creating the full Pipeline')

estimator=rndforest_clf
full_pipeline=Pipeline([('data_prep',dataprep_pipe),
                        ('model',rndforest_clf)])

full_pipeline.fit(X_train,y_train)

##Apply cross validation with the full_pipeline
cross_val_score(full_pipeline,X_train,y_train, cv=5, scoring='accuracy')


y_pred=full_pipeline.predict(X_test)

print ('Accuracy Score with',estimator,' estimator : ',accuracy_score(y_test, y_pred))
print('F1 Score with',estimator,' estimator : ',f1_score(y_test, y_pred, average='weighted'))
print('Precision Score with',estimator,' estimator : ',precision_score(y_test, y_pred, average='weighted'))
print('Recall Score with',estimator,' estimator : ',recall_score(y_test, y_pred, average='weighted'))
print('ROC_AUC score with',estimator,' estimator ', roc_auc_score(y_test, y_pred))

full_pipeline.get_params().keys()

#############################
##Step 5 GridSearchCV to find best params
#############################

param_grid={'model': [SGDClassifier(),LogisticRegression(),LinearSVC(),SVC(),DecisionTreeClassifier(),RandomForestClassifier()],
            'data_prep__numeric_pipe__data_missing__strategy':['median','mean','iterative','knn'],
            'data_prep__numeric_pipe__features_select__k_out_features': [1,2,3,4,5,6,7,8,9,10,11],
            'data_prep__numeric_pipe__features_select__rfe_estimator':['LogisticRegression','SVR'],
            'data_prep__numeric_pipe__features_select__strategy':['filter_num','filter_mutinf','wrapper_RFECV','wrapper_BackElim','LassoCV','RidgeCV'] ,
            'data_prep__category_pipe__data_missing__strategy': ['most_frequent','constant'],
            'data_prep__category_pipe__features_select__k_out_features': [1,2,3,4,5,6,7,8,9,10,11,12,13],
            'data_prep__category_pipe__features_select__rfe_estimator':['LogisticRegression','SVR'],
            'data_prep__category_pipe__features_select__strategy': ['filter_cat','filter_mutinf','wrapper_RFECV','wrapper_BackElim','LassoCV','RidgeCV'],
    }

from sklearn.model_selection import GridSearchCV
clf=GridSearchCV(full_pipeline,param_grid, cv=5)
clf.fit(X_train,y_train)
