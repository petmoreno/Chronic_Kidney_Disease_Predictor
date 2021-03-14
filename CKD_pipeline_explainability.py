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
from sklearn.model_selection import GridSearchCV

#Classifier models to use
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

import joblib

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
#Before starting to clean data, lets split train set and data set with stratrification on y=classification

train_set,test_set=train_test_split(df, test_size=0.3, random_state=42, stratify=df["classification"])


# from sklearn.model_selection import StratifiedShuffleSplit

# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(df, df["classification"]):
#     strat_train_set = df.loc[train_index]
#     strat_test_set = df.loc[test_index]
    
train_set['classification'].value_counts()
test_set['classification'].value_counts()
    

# strat_train_set['classification'].value_counts()
# strat_test_set['classification'].value_counts()

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
label_enc.classes_

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


#############################
##Step 4 Pipeline creation for model
#############################
#Several classifier with Cross validation will be applied
y_test=misspellingCorrector(y_test)

label_enc=LabelEncoder()
y_test=label_enc.fit_transform(y_test)

#Init the clasfifier
sgd_clf=SGDClassifier()
logreg_clf=LogisticRegression()
linsvc_clf=LinearSVC()
svc_clf=SVC()
dectree_clf=DecisionTreeClassifier(random_state=42)
rndforest_clf=RandomForestClassifier(random_state=42)
extratree_clf=ExtraTreesClassifier(random_state=42)
knn_clf=KNeighborsClassifier()
mlp_clf= MLPClassifier(alpha=1, max_iter=1000)
ada_clf= AdaBoostClassifier(random_state=42)
nb_clf= GaussianNB()
disc_clf=QuadraticDiscriminantAnalysis()
xgboost_clf= xgb.XGBClassifier(random_state=42)
gradboost_clf=GradientBoostingClassifier(random_state=42)

#
print ('Creating the full Pipeline')

estimator=rndforest_clf
full_pipeline=Pipeline([('data_prep',dataprep_pipe),
                        ('model',rndforest_clf)])


scoring=['accuracy','f1','precision','recall']
#############################
##Step 5 GridSearchCV to find best params
#############################

######v1_exp to test the best option found in the previous GridSearch with all classifiers


param_grid_v1_exp={'model': [logreg_clf,svc_clf,dectree_clf, rndforest_clf, knn_clf, mlp_clf, xgboost_clf, extratree_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [1,2,3,4,5,6,7],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num','wrapper_RFE'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [1,2,3,4,5,6,7],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat','wrapper_RFE']
     }
from sklearn.metrics import make_scorer
scoring2 = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0),
    'precision':make_scorer(precision_score),
    'f1':make_scorer(f1_score),
    'roc_auc':make_scorer(roc_auc_score)
}


#load model to save time of fitting
clf_v1_exp= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v1_exp.pkl')

clf_v1_exp=GridSearchCV(full_pipeline,param_grid_v1_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v1_exp.fit(X_train,y_train)
clf_v1_exp.best_estimator_
print('Params of best estimator of clf_v1_exp:', clf_v1_exp.best_params_)
#Results:Params of best estimator of clf_v10: {'data_prep__category_pipe__features_select__k_out_features': 5,
 # 'data_prep__category_pipe__features_select__strategy': 'wrapper_RFE', 
 # 'data_prep__numeric_pipe__data_missing__strategy': 'median',
 # 'data_prep__numeric_pipe__features_select__k_out_features':4 , 
 # 'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFE', 'model':AdaBoostClassifier()}

print('Score of best estimator of clf_v1_exp:', clf_v1_exp.best_score_)
#Score of best estimator of clf_v11: 1.0

print('Index of best estimator of clf_v1_exp:', clf_v1_exp.best_index_)
#Index of best estimator of clf_v11: 19

df_results_v1_exp=pd.DataFrame(clf_v1_exp.cv_results_)
df_results_v1_exp.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v1_exp.csv',index=False)
df_results_v1_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v1_exp.xlsx',index=False)
clf_v1_exp.refit
preds = clf_v1_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0,0.9583
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)

y_pred_1_exp=clf_v1_exp.predict(X_test)

#Saving the model
joblib.dump(clf_v1_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v1_exp.pkl', compress=1)



param_grid_v2_exp={'model': [rndforest_clf,  extratree_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [1,2,4,6,7],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num','wrapper_RFE'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [4,5,6,7],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat','wrapper_RFE']
     }

clf_v2_exp=GridSearchCV(full_pipeline,param_grid_v2_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v2_exp.fit(X_train,y_train)
clf_v2_exp.best_estimator_
print('Params of best estimator of clf_v2_exp:', clf_v2_exp.best_params_)
#Results:Params of best estimator of clf_v2_exp: {'data_prep__category_pipe__features_select__k_out_features': 7,
 # 'data_prep__category_pipe__features_select__strategy': 'wrapper_RFE', 
 # 'data_prep__numeric_pipe__data_missing__strategy': 'median',
 # 'data_prep__numeric_pipe__features_select__k_out_features':7 , 
 # 'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFE', 'model':AdaBoostClassifier()}

print('Score of best estimator of clf_v2_exp:', clf_v2_exp.best_score_)
#Score of best estimator of clf_v11: 1

print('Index of best estimator of clf_v2_exp:', clf_v2_exp.best_index_)
#Index of best estimator of clf_v11: 19

clf_v2_exp.refit
preds = clf_v2_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0,9916666666666667
y_pred_v2_exp = clf_v2_exp.predict(X_test)
df_results_v2_exp=pd.DataFrame(clf_v2_exp.cv_results_)
df_results_v2_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v2_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v2_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v2_exp.pkl', compress=1)


#Let's take the best estimator made with RandomForest
param_grid_v3_exp={'model': [rndforest_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [1],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [7],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat']
     }

clf_v3_exp=GridSearchCV(full_pipeline,param_grid_v3_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v3_exp.fit(X_train,y_train)
clf_v3_exp.best_estimator_
print('Params of best estimator of clf_v3_exp:', clf_v3_exp.best_params_)

print('Score of best estimator of clf_v3_exp:', clf_v3_exp.best_score_)
#Score of best estimator of clf_v11: 1

print('Index of best estimator of clf_v3_exp:', clf_v3_exp.best_index_)
#Index of best estimator of clf_v11: 19

clf_v3_exp.refit
preds = clf_v3_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0,9916666666666667
y_pred_v3_exp = clf_v3_exp.predict(X_test)
df_results_v3_exp=pd.DataFrame(clf_v3_exp.cv_results_)
df_results_v3_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v3_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v3_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v3_exp.pkl', compress=1)

#Let's take the best estimator made with XGBoost
param_grid_v4_exp={'model': [xgboost_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [3],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [2],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat']
     }

clf_v4_exp=GridSearchCV(full_pipeline,param_grid_v4_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v4_exp.fit(X_train,y_train)
clf_v4_exp.best_estimator_
print('Params of best estimator of clf_v4_exp:', clf_v4_exp.best_params_)

print('Score of best estimator of clf_v4_exp:', clf_v4_exp.best_score_)
#Score of best estimator of clf_v11: 0,9857142857142858

print('Index of best estimator of clf_v4_exp:', clf_v4_exp.best_index_)
#Index of best estimator of clf_v11: 19

clf_v4_exp.refit
preds = clf_v4_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0,958333
y_pred_v4_exp = clf_v4_exp.predict(X_test)
df_results_v4_exp=pd.DataFrame(clf_v4_exp.cv_results_)
df_results_v4_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v4_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v4_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v4_exp.pkl', compress=1)


#Let's take the best estimator made with extratree and 4 num feat and 6 cat feat
param_grid_v5_exp={'model': [extratree_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [4],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [6],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat']
     }

clf_v5_exp=GridSearchCV(full_pipeline,param_grid_v5_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v5_exp.fit(X_train,y_train)
clf_v5_exp.best_estimator_
print('Params of best estimator of clf_v5_exp:', clf_v5_exp.best_params_)

print('Score of best estimator of clf_v5_exp:', clf_v5_exp.best_score_)
#Score of best estimator of clf_v11: 0,9928571

print('Index of best estimator of clf_v5_exp:', clf_v5_exp.best_index_)
#Index of best estimator of clf_v11: 19

clf_v5_exp.refit
preds = clf_v5_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0,98333
y_pred_v5_exp = clf_v5_exp.predict(X_test)
df_results_v5_exp=pd.DataFrame(clf_v5_exp.cv_results_)
df_results_v5_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v4_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v5_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v4_exp.pkl', compress=1)

### Try gridsearch only with ensemble trees
param_grid_v6_exp={'model': [dectree_clf,rndforest_clf,extratree_clf,ada_clf,gradboost_clf,xgboost_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [1,2,3,4,5,6,7],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num','wrapper_RFE'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [1,2,3,4,5,6,7],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat','wrapper_RFE']
     }

#load model to save time of fitting
clf_v6_exp= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v6_exp.pkl')

clf_v6_exp=GridSearchCV(full_pipeline,param_grid_v6_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v6_exp.fit(X_train,y_train)
clf_v6_exp.best_estimator_
print('Params of best estimator of clf_v6_exp:', clf_v6_exp.best_params_)
#Results:Params of best estimator of clf_v10: {'data_prep__category_pipe__features_select__k_out_features': 5,
 # 'data_prep__category_pipe__features_select__strategy': 'wrapper_RFE', 
 # 'data_prep__numeric_pipe__data_missing__strategy': 'median',
 # 'data_prep__numeric_pipe__features_select__k_out_features':4 , 
 # 'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFE', 'model':AdaBoostClassifier()}

print('Score of best estimator of clf_v6_exp:', clf_v6_exp.best_score_)
#Score of best estimator of clf_v11: 1.0

print('Index of best estimator of clf_v6_exp:', clf_v6_exp.best_index_)
#Index of best estimator of clf_v11: 19

df_results_v6_exp=pd.DataFrame(clf_v6_exp.cv_results_)
df_results_v6_exp.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v6_exp.csv',index=False)
df_results_v6_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v6_exp.xlsx',index=False)
clf_v6_exp.refit
preds = clf_v6_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.9916666666666667
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)

y_pred_6_exp=clf_v6_exp.predict(X_test)

#Saving the model
joblib.dump(clf_v6_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v6_exp.pkl', compress=1)

##Dectree best estimator

#load model to save time of fitting
clf_v61_exp= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v61_exp.pkl')

param_grid_v61_exp={'model': [dectree_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [3],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [4],           
             'data_prep__category_pipe__features_select__strategy': ['wrapper_RFE']
     }

#load model to save time of fitting
clf_v61_exp=GridSearchCV(full_pipeline,param_grid_v61_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v61_exp.fit(X_train,y_train)

df_results_v61_exp=pd.DataFrame(clf_v61_exp.cv_results_)
df_results_v61_exp.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v61_exp.csv',index=False)
df_results_v61_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v61_exp.xlsx',index=False)
clf_v61_exp.refit
preds = clf_v61_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.95
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)

y_pred_v61_exp=clf_v61_exp.predict(X_test)

#Saving the model
joblib.dump(clf_v61_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v61_exp.pkl', compress=1)

##AdaBoost best estimator
param_grid_v62_exp={'model': [ada_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [4],            
             'data_prep__numeric_pipe__features_select__strategy':['wrapper_RFE'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [5],           
             'data_prep__category_pipe__features_select__strategy': ['wrapper_RFE']
     }

#load model to save time of fitting
clf_v62_exp=GridSearchCV(full_pipeline,param_grid_v62_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v62_exp.fit(X_train,y_train)

df_results_v62_exp=pd.DataFrame(clf_v62_exp.cv_results_)
df_results_v62_exp.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v62_exp.csv',index=False)
df_results_v62_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v62_exp.xlsx',index=False)
clf_v62_exp.refit
preds = clf_v62_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.983333333
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)

y_pred_v62_exp=clf_v62_exp.predict(X_test)

#Saving the model
joblib.dump(clf_v62_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v62_exp.pkl', compress=1)

##ExtraTree best estimator
clf_v63_exp= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v63_exp.pkl')

param_grid_v63_exp={'model': [extratree_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [4],            
             'data_prep__numeric_pipe__features_select__strategy':['wrapper_RFE'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [4],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat']
     }

#load model to save time of fitting
clf_v63_exp=GridSearchCV(full_pipeline,param_grid_v63_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v63_exp.fit(X_train,y_train)

df_results_v63_exp=pd.DataFrame(clf_v63_exp.cv_results_)
df_results_v63_exp.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v63_exp.csv',index=False)
df_results_v63_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v63_exp.xlsx',index=False)
clf_v63_exp.refit
preds = clf_v63_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.9916666666666667
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)

y_pred_v63_exp=clf_v63_exp.predict(X_test)

#Saving the model
joblib.dump(clf_v63_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v63_exp.pkl', compress=1)

##gradboost_clf best estimator
param_grid_v64_exp={'model': [gradboost_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [2],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [4],           
             'data_prep__category_pipe__features_select__strategy': ['wrapper_RFE']
     }

#load model to save time of fitting
clf_v64_exp=GridSearchCV(full_pipeline,param_grid_v64_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v64_exp.fit(X_train,y_train)

df_results_v64_exp=pd.DataFrame(clf_v64_exp.cv_results_)
df_results_v64_exp.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v64_exp.csv',index=False)
df_results_v64_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v64_exp.xlsx',index=False)
clf_v64_exp.refit
preds = clf_v64_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.9666666666666667
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)

y_pred_v64_exp=clf_v64_exp.predict(X_test)

#Saving the model
joblib.dump(clf_v64_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v64_exp.pkl', compress=1)

##rnd_forest best estimator
param_grid_v65_exp={'model': [rndforest_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [1],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [7],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat']
     }

#load model to save time of fitting
clf_v65_exp=GridSearchCV(full_pipeline,param_grid_v65_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v65_exp.fit(X_train,y_train)

df_results_v65_exp=pd.DataFrame(clf_v65_exp.cv_results_)
df_results_v65_exp.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v65_exp.csv',index=False)
df_results_v65_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v65_exp.xlsx',index=False)
clf_v65_exp.refit
preds = clf_v65_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.9833333333333333
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)

y_pred_v65_exp=clf_v65_exp.predict(X_test)

#Saving the model
joblib.dump(clf_v65_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v65_exp.pkl', compress=1)

##xg_boost best estimator
param_grid_v66_exp={'model': [xgboost_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [3],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [2],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat']
     }

#load model to save time of fitting
clf_v66_exp=GridSearchCV(full_pipeline,param_grid_v66_exp,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v66_exp.fit(X_train,y_train)

df_results_v66_exp=pd.DataFrame(clf_v66_exp.cv_results_)
df_results_v66_exp.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v66_exp.csv',index=False)
df_results_v66_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v66_exp.xlsx',index=False)
clf_v66_exp.refit
preds = clf_v66_exp.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.9583333333333334
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)

y_pred_v66_exp=clf_v66_exp.predict(X_test)

#Saving the model
joblib.dump(clf_v66_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v66_exp.pkl', compress=1)

#########################
#Lets build a dataframe with the results of all best_stimator in the different grid search and with the prediction results in X_tests of all best_estimators in the different grid search
# #######################

overall_results={'clf':['clf_v2_exp','clf_v3_exp','clf_v4_exp'],
                 'params':[clf_v2_exp.best_params_, clf_v3_exp.best_params_, clf_v4_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v2_exp),accuracy_score(y_test, y_pred_v3_exp),accuracy_score(y_test, y_pred_v4_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v2_exp, average='weighted'), f1_score(y_test, y_pred_v3_exp, average='weighted'), f1_score(y_test, y_pred_v4_exp, average='weighted')],
                 'precision_test':[precision_score(y_test, y_pred_v2_exp, average='weighted'),precision_score(y_test, y_pred_v3_exp, average='weighted'),precision_score(y_test, y_pred_v4_exp, average='weighted')],
                 'recall_test':[recall_score(y_test, y_pred_v2_exp, average='weighted'),recall_score(y_test, y_pred_v3_exp, average='weighted'),recall_score(y_test, y_pred_v4_exp, average='weighted')],
                 'specificity_test':[recall_score(y_test, y_pred_v2_exp, average='weighted',pos_label=0),recall_score(y_test, y_pred_v3_exp, average='weighted',pos_label=0),recall_score(y_test, y_pred_v4_exp, average='weighted',pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v2_exp),roc_auc_score(y_test, y_pred_v3_exp),roc_auc_score(y_test, y_pred_v4_exp)]    
    }

# df_overall_results=pd.DataFrame(data=overall_results)
# df_overall_results.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_overall_results.xlsx',index=False)

df_overall_results_paper=pd.DataFrame(data=overall_results)
df_overall_results_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_overall_results_paper_v2_exp.xlsx',index=False)

#Lets calculate the confusion matrix of the best 3 results of GridSearch
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_v2_exp)

confusion_matrix(y_test, y_pred_v3_exp)

confusion_matrix(y_test, y_pred_v4_exp)

#########################
#Lets build a dataframe with the results of ensemble trees best_stimator in the different grid search and with the prediction results in X_tests 


overall_results={'clf':['clf_v61_exp','clf_v62_exp','clf_v63_exp','clf_v64_exp','clf_v65_exp','clf_v66_exp'],
                 'params':[clf_v61_exp.best_params_, clf_v62_exp.best_params_, clf_v63_exp.best_params_, clf_v64_exp.best_params_,clf_v65_exp.best_params_,clf_v66_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v61_exp),accuracy_score(y_test, y_pred_v62_exp),accuracy_score(y_test, y_pred_v63_exp),accuracy_score(y_test, y_pred_v64_exp),accuracy_score(y_test, y_pred_v65_exp),accuracy_score(y_test, y_pred_v66_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v61_exp, average='weighted'), f1_score(y_test, y_pred_v62_exp, average='weighted'), f1_score(y_test, y_pred_v63_exp, average='weighted'),f1_score(y_test, y_pred_v64_exp, average='weighted'),f1_score(y_test, y_pred_v65_exp, average='weighted'),f1_score(y_test, y_pred_v66_exp, average='weighted')],
                 'precision_test':[precision_score(y_test, y_pred_v61_exp, average='weighted'),precision_score(y_test, y_pred_v62_exp, average='weighted'),precision_score(y_test, y_pred_v63_exp, average='weighted'),precision_score(y_test, y_pred_v64_exp, average='weighted'),precision_score(y_test, y_pred_v65_exp, average='weighted'),precision_score(y_test, y_pred_v66_exp, average='weighted')],
                 'recall_test':[recall_score(y_test, y_pred_v61_exp, average='weighted'),recall_score(y_test, y_pred_v62_exp, average='weighted'),recall_score(y_test, y_pred_v63_exp, average='weighted'),recall_score(y_test, y_pred_v64_exp, average='weighted'),recall_score(y_test, y_pred_v65_exp, average='weighted'),recall_score(y_test, y_pred_v66_exp, average='weighted')],
                 'specificity_test':[recall_score(y_test, y_pred_v61_exp, average='weighted',pos_label=0),recall_score(y_test, y_pred_v62_exp, average='weighted',pos_label=0),recall_score(y_test, y_pred_v63_exp, average='weighted',pos_label=0),recall_score(y_test, y_pred_v64_exp, average='weighted',pos_label=0),recall_score(y_test, y_pred_v65_exp, average='weighted',pos_label=0),recall_score(y_test, y_pred_v66_exp, average='weighted',pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v61_exp),roc_auc_score(y_test, y_pred_v62_exp),roc_auc_score(y_test, y_pred_v63_exp),roc_auc_score(y_test, y_pred_v64_exp),roc_auc_score(y_test, y_pred_v65_exp),roc_auc_score(y_test, y_pred_v66_exp)]    
    }

# df_overall_results=pd.DataFrame(data=overall_results)
# df_overall_results.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_overall_results.xlsx',index=False)

df_overall_results_paper=pd.DataFrame(data=overall_results)
df_overall_results_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_overall_results_paper_exp_ensembleTrees.xlsx',index=False)

#Lets calculate the confusion matrix of the best 3 results of GridSearch
from sklearn.metrics import confusion_matrix
#decision tree
print('confusion matrix of decision tree',confusion_matrix(y_test, y_pred_v61_exp))
print('confusion matrix of adaboost',confusion_matrix(y_test, y_pred_v62_exp))
print('confusion matrix of extratree',confusion_matrix(y_test, y_pred_v63_exp))
print('confusion matrix of graboost',confusion_matrix(y_test, y_pred_v64_exp))
print('confusion matrix of rndforest',confusion_matrix(y_test, y_pred_v65_exp))
print('confusion matrix of xgboost',confusion_matrix(y_test, y_pred_v66_exp))


###########Features selected

pipe_numeric_featsel= Pipeline([('mispelling',adhoc_transf.misspellingTransformer()),
                                 ('features_cast',adhoc_transf.Numeric_Cast_Column()),
                                 ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median'))
                                 
                        ])

pipe_category_feat= Pipeline([('mispelling',adhoc_transf.misspellingTransformer()),
                                 ('features_cast',adhoc_transf.Category_Cast_Column()),
                                 ('data_missing',missing_val_imput.Category_Imputer(strategy='most_frequent')),
                                 ('cat_feat_engineering',adhoc_transf.CastDown()),
                                 ('encoding', OrdinalEncoder())
                        ])

X_train_numfeatsel=pipe_numeric_featsel.fit_transform(X_train[numerical_features])
df_X_train_numfeatsel=pd.DataFrame(X_train_numfeatsel, columns=numerical_features)

X_train_catfeatsel=pipe_category_feat.fit_transform(X_train[category_features])
df_X_train_catfeatsel=pd.DataFrame(X_train_catfeatsel, columns=category_features)

# numerical_features=['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
# category_features= ['sg','al','su','rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']


###clf v63_exp: ExtraTrees, median, numRFE-4,nomChiSquared-4 
feature_select.feat_sel_RFE(df_X_train_numfeatsel,y_train,k_out_features=4)
#numerical_features_v63_exp=['sc','pot','hemo','rc']
feature_select.feat_sel_Cat_to_Cat_chi2(df_X_train_catfeatsel,y_train,k_out_features=4)
# category_features_v63_exp= ['al','su','htn', 'dm']

###clf v62_exp: Adaboost, median, numRFE-4,catRFE-5 
feature_select.feat_sel_RFE(df_X_train_numfeatsel,y_train,k_out_features=4)
#numerical_features_v62_exp=['sc','pot','hemo','rc']

feature_select.feat_sel_RFE(df_X_train_catfeatsel,y_train,k_out_features=5)
# category_features_v62_exp= ['sg','al','htn', 'dm','pe']

###clf v4_exp: XGBoost, median, numANOVA-3,nomChiSquared-2 
feature_select.feat_sel_Num_to_Cat(df_X_train_numfeatsel,y_train,k_out_features=3)
#numerical_features_v3_exp=['hemo','pcv','rc']

feature_select.feat_sel_Cat_to_Cat_chi2(df_X_train_catfeatsel,y_train,k_out_features=2)
# category_features_v3_exp= ['al','su']

###############Features importance of best estimator with test set
#The best estimator with the test set is clf_v12, therefore let's explore the feature importance of it
#
win_model=clf_v12.best_estimator_
clf_v12.best_params_
# {'data_prep__category_pipe__features_select__k_out_features': 5,
#  'data_prep__category_pipe__features_select__strategy': 'wrapper_RFE',
#  'data_prep__numeric_pipe__data_missing__strategy': 'median',
#  'data_prep__numeric_pipe__features_select__k_out_features': 7,
#  'data_prep__numeric_pipe__features_select__strategy': 'filter_num',
#  'model': AdaBoostClassifier()}


# Creating a Pipeline for showing the feature importance
features_selected=['sc','pot','hemo','rc','sg','al','su','htn', 'dm', 'appet']
X_train_feat_selected=X_train[features_selected]
num_feat_selected=['sc','pot','hemo','rc']
cat_feat_selected= ['sg','al','su','htn', 'dm', 'appet']



feat_sel_pipe=ColumnTransformer([('numeric_pipe',pipe_numeric_featsel,num_feat_selected),
                                 ('category_pipe',pipe_category_feat, cat_feat_selected)
                                ])

X_train_featsel=feat_sel_pipe.fit_transform(X_train_feat_selected)
df_X_train_featsel=pd.DataFrame(X_train_featsel, columns=features_selected)

extratree_clf.fit(df_X_train_featsel,y_train)
extratree_clf.feature_importances_

win_model.named_steps['model'].feature_importances_

#Several solution to plot importances
#Solution 1
feat_importances = pd.Series(ada_clf.feature_importances_, index=df_X_train_featsel.columns)
feat_importances.nlargest(20).plot(kind='barh')

#Solution 2
features = features_selected
importances = extratree_clf.feature_importances_
indices = np.argsort(importances)

plt.title('Features Importance with 10 features selected')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Gini Importance')
plt.show()

#############################################################
#We take the test model with least features selected #9
features_selected_9=['hemo','sg','al','su','htn', 'dm', 'appet','pe']
X_train_feat_selected_9=X_train[features_selected_9]
num_feat_selected_9=['hemo']
cat_feat_selected_9= ['sg','al','su','htn', 'dm', 'appet','pe']



feat_sel_pipe=ColumnTransformer([('numeric_pipe',pipe_numeric_featsel,num_feat_selected_9),
                                 ('category_pipe',pipe_category_feat, cat_feat_selected_9)
                                ])

X_train_featsel_9=feat_sel_pipe.fit_transform(X_train_feat_selected_9)
df_X_train_featsel_9=pd.DataFrame(X_train_featsel_9, columns=features_selected_9)


extratree_clf.fit(X_train_featsel_9,y_train)
extratree_clf.feature_importances_

#Several solution to plot importances

#Solution 2
features = features_selected_9
importances = extratree_clf.feature_importances_
indices = np.argsort(importances)

plt.title('Features Importance with 8 features selected')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Gini Importance')
plt.show()

#############################################################
#We take the test model with least features selected #9
features_selected_5=['hemo','pcv','rc','al','su']
X_train_feat_selected_5=X_train[features_selected_5]
num_feat_selected_5=['hemo','pcv','rc']
cat_feat_selected_5= ['al','su']



feat_sel_pipe=ColumnTransformer([('numeric_pipe',pipe_numeric_featsel,num_feat_selected_5),
                                 ('category_pipe',pipe_category_feat, cat_feat_selected_5)
                                ])

X_train_featsel_5=feat_sel_pipe.fit_transform(X_train_feat_selected_5)
df_X_train_featsel_5=pd.DataFrame(X_train_featsel_5, columns=features_selected_5)


extratree_clf.fit(X_train_featsel_5,y_train)
extratree_clf.feature_importances_

#Several solution to plot importances

#Solution 2
features = features_selected_5
importances = extratree_clf.feature_importances_
indices = np.argsort(importances)

plt.title('Features Importance with 5 features selected')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Gini Importance')
plt.show()

#####################################################################
#We take the best model for the paper publication Extratrees with RFE-4 numeric and ChiSquared-4 nominal


clf_v63_exp.best_params_
# {'data_prep__category_pipe__features_select__k_out_features': 4,
#  'data_prep__category_pipe__features_select__strategy': 'filter_cat',
#  'data_prep__numeric_pipe__data_missing__strategy': 'median',
#  'data_prep__numeric_pipe__features_select__k_out_features': 4,
#  'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFE',
#  'model': ExtraTreesClassifier(random_state=42)}

features_selected_clf_v63_exp=['sc','pot','hemo','rc','al','su','htn', 'dm']
X_train_feat_selected_clf_v63_exp=X_train[features_selected_clf_v63_exp]
num_feat_selected_clf_v63_exp=['sc','pot','hemo','rc']
cat_feat_selected_clf_v63_exp= ['al','su','htn', 'dm']



feat_sel_pipe=ColumnTransformer([('numeric_pipe',pipe_numeric_featsel,num_feat_selected_clf_v63_exp),
                                 ('category_pipe',pipe_category_feat, cat_feat_selected_clf_v63_exp)
                                ])

X_train_featsel_clf_v63_exp=feat_sel_pipe.fit_transform(X_train_feat_selected_clf_v63_exp)
df_X_train_featsel_clf_v63_exp=pd.DataFrame(X_train_featsel_clf_v63_exp, columns=features_selected_clf_v63_exp)


extratree_clf.fit(df_X_train_featsel_clf_v63_exp,y_train)
extratree_clf.feature_importances_

#Several solution to plot importances

#Solution 2
features = features_selected_clf_v63_exp
importances = extratree_clf.feature_importances_
indices = np.argsort(importances)

plt.title('Features Importance of ExtraTrees with 8 features selected')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Gini Importance')
plt.show()

#####################
#Building the DT
clf_tree_feat=DecisionTreeClassifier(max_depth=3, random_state=42)
#use grid_search_CV to see the results
clf_tree_feat.get_params().keys()

param_grid_clf_tree_feat={'max_depth': [3,None]}

clf_tree_feat_grid=GridSearchCV(clf_tree_feat,param_grid_clf_tree_feat,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_tree_feat_grid.fit(df_X_train_featsel_clf_v63_exp,y_train)
clf_tree_feat_grid.best_estimator_
clf_tree_feat_grid.best_params_
df_results_clf_tree_feat=pd.DataFrame(clf_tree_feat_grid.cv_results_)
# create an excel with the cross val resutls
df_results_clf_tree_feat.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_tree_feat.xlsx',index=False)
#Fit the best estimator with the test set
clf_tree_feat_grid.refit

X_test_feat_selected_clf_v63_exp=X_test[features_selected_clf_v63_exp]
num_feat_selected_clf_v63_exp=['sc','pot','hemo','rc']
cat_feat_selected_clf_v63_exp= ['al','su','htn', 'dm']



feat_sel_pipe=ColumnTransformer([('numeric_pipe',pipe_numeric_featsel,num_feat_selected_clf_v63_exp),
                                 ('category_pipe',pipe_category_feat, cat_feat_selected_clf_v63_exp)
                                ])

X_test_featsel_clf_v63_exp=feat_sel_pipe.fit_transform(X_test_feat_selected_clf_v63_exp)
df_X_test_featsel_clf_v63_exp=pd.DataFrame(X_test_featsel_clf_v63_exp, columns=features_selected_clf_v63_exp)

preds = clf_tree_feat_grid.predict(df_X_test_featsel_clf_v63_exp)
np.mean(preds == y_test)#0.9583333333333334
y_pred_tree=clf_tree_feat_grid.predict(df_X_test_featsel_clf_v63_exp)

overall_test_results_clf_tree_feat={'clf':['clf_tree_feat'],
                 'params':[clf_tree_feat_grid.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_tree)],
                 'f1_test':[f1_score(y_test, y_pred_tree, average='weighted')],
                 'precision_test':[precision_score(y_test, y_pred_tree, average='weighted')],
                 'recall_test':[recall_score(y_test, y_pred_tree, average='weighted')],
                 'specificity_test':[recall_score(y_test, y_pred_tree, pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_tree)]
    }
#Saving results with test set to calculate interpretability measures
df_overall_test_results_clf_tree_feat=pd.DataFrame(data=overall_test_results_clf_tree_feat)
df_overall_test_results_clf_tree_feat.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\overall_test_results_clf_tree_feat.xlsx',index=False)

##############################
#Lets print the decision tree
from sklearn import tree
import graphviz
y_train_df=pd.DataFrame(y_train)

dot_data = tree.export_graphviz(clf_tree_feat_grid.best_estimator_, 
                  feature_names=X_train_feat_selected_clf_v63_exp.columns,  
                  class_names=['ckd','not_ckd'],  
                  filled=True, rounded=True,  
                  special_characters=True,
                   out_file=None,
                           )

graph = graphviz.Source(dot_data)
graph.format = "png"
graph.render(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\dt_featsel_graph.')

#the decisiontree shows cut point in the branches with the following values
#time=0.254, 0.175
#ejection_fraction=0.115, 0.192
#serum creatinine=0.107
#To improve readability of the graph lets transform these values to their original ones
#Inverting time
X_train_feat_time=X_train[['time']].copy()
minmaxtrain_time=MinMaxScaler()
X_train_feat_time=minmaxtrain_time.fit_transform(X_train_feat_time)
time_1=minmaxtrain_time.inverse_transform(np.array(0.254).reshape(1,-1))
time_2=minmaxtrain_time.inverse_transform(np.array(0.175).reshape(1,-1))
#Inverting ejection_fraction
X_train_feat_ef=X_train[['ejection_fraction']].copy()
minmaxtrain_ef=MinMaxScaler()
X_train_feat_ef=minmaxtrain_ef.fit_transform(X_train_feat_ef)
ef_1=minmaxtrain_ef.inverse_transform(np.array(0.115).reshape(1,-1))
ef_2=minmaxtrain_ef.inverse_transform(np.array(0.192).reshape(1,-1))
#Inverting serum_creatinine
X_train_feat_sc=X_train[['serum_creatinine']].copy()
minmaxtrain_sc=MinMaxScaler()
X_train_feat_sc=minmaxtrain_sc.fit_transform(X_train_feat_sc)
sc_1=minmaxtrain_sc.inverse_transform(np.array(0.107).reshape(1,-1))
print ('time_1: ',time_1,'; time_2: ',time_2,'; ef_1: ',ef_1,'; ef_2: ',ef_2,'; sc_1: ',sc_1)
