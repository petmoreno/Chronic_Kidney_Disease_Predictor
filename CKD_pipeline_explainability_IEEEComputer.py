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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier, VotingClassifier
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import joblib

#%matplotlib inline 


#importing file into a pandas dataframe# As being unable to extract data from it original source, the csv file is downloaded from
#https://www.kaggle.com/mansoordaku/ckdisease
path_data=r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease\Chronic_Kidney_Disease\kidney_disease.csv'
df=pd.read_csv(path_data)
df.head()

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

    
train_set['classification'].value_counts()
test_set['classification'].value_counts()

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

#Ordinal encoder user might be wrong since most of the category feature are not ordinal

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
dectree_clf=DecisionTreeClassifier(random_state=42)
rndforest_clf=RandomForestClassifier(random_state=42)
extratree_clf=ExtraTreesClassifier(random_state=42)
ada_clf= AdaBoostClassifier(random_state=42)
xgboost_clf= xgb.XGBClassifier(random_state=42)
gradboost_clf=GradientBoostingClassifier(random_state=42)
voting_clf=VotingClassifier(estimators=[('rdf', rndforest_clf), ('xtra', extratree_clf), ('ada', ada_clf)], voting='soft')
#
print ('Creating the full Pipeline')

full_pipeline=Pipeline([('data_prep',dataprep_pipe),
                        ('model',rndforest_clf)])


#Init the scorer
from sklearn.metrics import make_scorer
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0),
    'precision':make_scorer(precision_score),
    'f1':make_scorer(f1_score),
    'roc_auc':make_scorer(roc_auc_score)
}

#############################
##Step 5 GridSearchCV to find best params
#############################

######v1_exp to test the best option found in the previous GridSearch with all classifiers


param_grid_v1_exp={'model': [dectree_clf, rndforest_clf, extratree_clf, ada_clf, xgboost_clf,gradboost_clf,voting_clf ],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [1,2,3,4,5,6,7],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num','wrapper_RFE'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [1,2,3,4,5,6,7],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat','wrapper_RFE']
     }



#load model to save time of fitting
clf_v1_exp= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\clf_v1_exp.pkl')

clf_v1_exp=GridSearchCV(full_pipeline,param_grid_v1_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v1_exp.fit(X_train,y_train)

#Saving the model
joblib.dump(clf_v1_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\clf_v1_exp.pkl', compress=1)

clf_v1_exp.best_estimator_
print('Params of best estimator of clf_v1_exp:', clf_v1_exp.best_params_)
#Results:Params of best estimator of clf_v10: {'data_prep__category_pipe__features_select__k_out_features': 4,
 # 'data_prep__category_pipe__features_select__strategy': 'filter_cat', 
 # 'data_prep__numeric_pipe__data_missing__strategy': 'median',
 # 'data_prep__numeric_pipe__features_select__k_out_features':4 , 
 # 'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFE', 'model':ExtraTreesClassifier()}

print('Score of best estimator of clf_v1_exp:', clf_v1_exp.best_score_)
#Score of best estimator of clf_v11: 1.0

print('Index of best estimator of clf_v1_exp:', clf_v1_exp.best_index_)
#Index of best estimator of clf_v11: 639

df_results_v1_exp=pd.DataFrame(clf_v1_exp.cv_results_)
df_results_v1_exp.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\df_results_v1_exp.csv',index=False)
df_results_v1_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\df_results_v1_exp.xlsx',index=False)
clf_v1_exp.refit
y_pred_v1_exp = clf_v1_exp.predict(X_test)
np.mean(y_pred_v1_exp == y_test)#0.9916666666666667
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_v1_exp)

#########################
#Lets build a dataframe with the results of all best_stimator in the different grid search and with the prediction results in X_tests of all best_estimators in the different grid search
# #######################

overall_results={'clf':['clf_v1_exp'],
                 'params':[clf_v1_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v1_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v1_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v1_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v1_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v1_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v1_exp)]    
    }

df_overall_results_paper=pd.DataFrame(data=overall_results)
df_overall_results_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\df_test_results_v1_exp.xlsx',index=False)


#######################
#Build a param grid for the best combination per each estimator considered
#v2:DecisionTree
#######################

param_grid_v2_exp={'model': [dectree_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [3],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [4],           
             'data_prep__category_pipe__features_select__strategy': ['wrapper_RFE']
     }

clf_v2_exp=GridSearchCV(full_pipeline,param_grid_v2_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v2_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v2_exp:', clf_v2_exp.best_score_) #Score of best estimator of clf_v2: 0.9928571428571429

#Saving the results in an excel
df_results_v2_exp=pd.DataFrame(clf_v2_exp.cv_results_)
df_results_v2_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\df_results_v2_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v2_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\clf_v2_exp.pkl', compress=1)

#Obtaining classification  with test set
clf_v2_exp.refit
y_pred_v2_exp = clf_v2_exp.predict(X_test)
np.mean(y_pred_v2_exp == y_test)#0,9916666666666667

test_results_DT={'clf':['clf_v2_exp'],
                 'params':[clf_v2_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v2_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v2_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v2_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v2_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v2_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v2_exp)]    
    }

test_results_DT_paper=pd.DataFrame(data=test_results_DT)
test_results_DT_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\test_results_DT_paper.xlsx',index=False)

######################
#v3:Random Forest
#######################

param_grid_v3_exp={'model': [rndforest_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [1],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [7],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat']
     }

clf_v3_exp=GridSearchCV(full_pipeline,param_grid_v3_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v3_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v3_exp:', clf_v3_exp.best_score_) #Score of best estimator of clf_v3: 1

#Saving the results in an excel
df_results_v3_exp=pd.DataFrame(clf_v3_exp.cv_results_)
df_results_v3_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\df_results_v3_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v3_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\clf_v3_exp.pkl', compress=1)

#Obtaining classification  with test set
clf_v3_exp.refit
y_pred_v3_exp = clf_v3_exp.predict(X_test)
np.mean(y_pred_v3_exp == y_test)#0.9833333333333333

test_results_RF={'clf':['clf_v3_exp'],
                 'params':[clf_v3_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v3_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v3_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v3_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v3_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v3_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v3_exp)]    
    }

test_results_RF_paper=pd.DataFrame(data=test_results_RF)
test_results_RF_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\test_results_RF_paper.xlsx',index=False)

######################
#v4:Extra Trees
#######################

param_grid_v4_exp={'model': [extratree_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [4],            
             'data_prep__numeric_pipe__features_select__strategy':['wrapper_RFE'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [4],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat']
     }

clf_v4_exp=GridSearchCV(full_pipeline,param_grid_v4_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v4_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v4_exp:', clf_v4_exp.best_score_) #Score of best estimator of clf_v4: 1

#Saving the results in an excel
df_results_v4_exp=pd.DataFrame(clf_v4_exp.cv_results_)
df_results_v4_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\df_results_v4_exp.xlsx',index=False)
#Saving the model
#joblib.dump(clf_v4_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\clf_v4_exp.pkl', compress=1)

#Obtaining classification  with test set
clf_v4_exp.refit
y_pred_v4_exp = clf_v4_exp.predict(X_test)
np.mean(y_pred_v4_exp == y_test)#0,9916666666666667

test_results_ET={'clf':['clf_v4_exp'],
                 'params':[clf_v4_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v4_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v4_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v4_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v4_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v4_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v4_exp)]    
    }

test_results_ET_paper=pd.DataFrame(data=test_results_ET)
test_results_ET_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\test_results_ET_paper.xlsx',index=False)

######################
#v5:AdaBoost
#######################

param_grid_v5_exp={'model': [ada_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [4],            
             'data_prep__numeric_pipe__features_select__strategy':['wrapper_RFE'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [5],           
             'data_prep__category_pipe__features_select__strategy': ['wrapper_RFE']
     }

clf_v5_exp=GridSearchCV(full_pipeline,param_grid_v5_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v5_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v5_exp:', clf_v5_exp.best_score_) #Score of best estimator of clf_v5: 1

#Saving the results in an excel
df_results_v5_exp=pd.DataFrame(clf_v5_exp.cv_results_)
df_results_v5_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\df_results_v5_exp.xlsx',index=False)
#Saving the model
#joblib.dump(clf_v5_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\clf_v5_exp.pkl', compress=1)

#Obtaining classification  with test set
clf_v5_exp.refit
y_pred_v5_exp = clf_v5_exp.predict(X_test)
np.mean(y_pred_v5_exp == y_test)#0.9833333333333333

test_results_AB={'clf':['clf_v5_exp'],
                 'params':[clf_v5_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v5_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v5_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v5_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v5_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v5_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v5_exp)]    
    }

test_results_AB_paper=pd.DataFrame(data=test_results_AB)
test_results_AB_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\test_results_AB_paper.xlsx',index=False)

######################
#v6:Gradient Boosting
#######################

param_grid_v6_exp={'model': [gradboost_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [2],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [4],           
             'data_prep__category_pipe__features_select__strategy': ['wrapper_RFE']
     }

clf_v6_exp=GridSearchCV(full_pipeline,param_grid_v6_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v6_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v6_exp:', clf_v6_exp.best_score_) #Score of best estimator of clf_v6: 0.9928571428571429

#Saving the results in an excel
df_results_v6_exp=pd.DataFrame(clf_v6_exp.cv_results_)
df_results_v6_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\df_results_v6_exp.xlsx',index=False)
#Saving the model
#joblib.dump(clf_v6_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\clf_v6_exp.pkl', compress=1)

#Obtaining classification  with test set
clf_v6_exp.refit
y_pred_v6_exp = clf_v6_exp.predict(X_test)
np.mean(y_pred_v6_exp == y_test)#0.9666666666666667

test_results_GB={'clf':['clf_v6_exp'],
                 'params':[clf_v6_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v6_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v6_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v6_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v6_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v6_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v6_exp)]    
    }

test_results_GB_paper=pd.DataFrame(data=test_results_GB)
test_results_GB_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\test_results_GB_paper.xlsx',index=False)

######################
#v7:eXtreme Gradient Boosting
#######################

param_grid_v7_exp={'model': [xgboost_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [3],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [2],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat']
     }

clf_v7_exp=GridSearchCV(full_pipeline,param_grid_v7_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v7_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v7_exp:', clf_v7_exp.best_score_) #Score of best estimator of clf_v7: 0.9857142857142858

#Saving the results in an excel
df_results_v7_exp=pd.DataFrame(clf_v7_exp.cv_results_)
df_results_v7_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\df_results_v7_exp.xlsx',index=False)
#Saving the model
#joblib.dump(clf_v7_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\clf_v7_exp.pkl', compress=1)

#Obtaining classification  with test set
clf_v7_exp.refit
y_pred_v7_exp = clf_v7_exp.predict(X_test)
np.mean(y_pred_v7_exp == y_test)#0.9583333333333334

test_results_XGB={'clf':['clf_v7_exp'],
                 'params':[clf_v7_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v7_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v7_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v7_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v7_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v7_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v7_exp)]    
    }

test_results_XGB_paper=pd.DataFrame(data=test_results_XGB)
test_results_XGB_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\test_results_XGB_paper.xlsx',index=False)

######################
#v8:Max Voting
#######################

param_grid_v8_exp={'model': [voting_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [1],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [7],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat']
     }

clf_v8_exp=GridSearchCV(full_pipeline,param_grid_v8_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v8_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v8_exp:', clf_v8_exp.best_score_) #Score of best estimator of clf_v8: 1

#Saving the results in an excel
df_results_v8_exp=pd.DataFrame(clf_v8_exp.cv_results_)
df_results_v8_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\df_results_v8_exp.xlsx',index=False)
#Saving the model
#joblib.dump(clf_v8_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\clf_v8_exp.pkl', compress=1)

#Obtaining classification  with test set
clf_v8_exp.refit
y_pred_v8_exp = clf_v8_exp.predict(X_test)
np.mean(y_pred_v8_exp == y_test)#0.9833333333333333

test_results_MaxV={'clf':['clf_v8_exp'],
                 'params':[clf_v8_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v8_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v8_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v8_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v8_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v8_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v8_exp)]    
    }

test_results_MaxV_paper=pd.DataFrame(data=test_results_MaxV)
test_results_MaxV_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\test_results_MaxV_paper.xlsx',index=False)


#########################
#Dataframe with the best estimator per each classifier applied to the Xtest
# #######################

overall_results={'clf':['clf_v1_exp','clf_v2_exp','clf_v3_exp','clf_v4_exp','clf_v5_exp','clf_v6_exp','clf_v7_exp','clf_v8_exp'],
                 'params':[clf_v1_exp.best_params_, clf_v2_exp.best_params_, clf_v3_exp.best_params_, clf_v4_exp.best_params_,clf_v5_exp.best_params_,clf_v6_exp.best_params_,clf_v7_exp.best_params_,clf_v8_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v1_exp),accuracy_score(y_test, y_pred_v2_exp),accuracy_score(y_test, y_pred_v3_exp),accuracy_score(y_test, y_pred_v4_exp),accuracy_score(y_test, y_pred_v5_exp),accuracy_score(y_test, y_pred_v6_exp),accuracy_score(y_test, y_pred_v7_exp),accuracy_score(y_test, y_pred_v8_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v1_exp),recall_score(y_test, y_pred_v2_exp),recall_score(y_test, y_pred_v3_exp),recall_score(y_test, y_pred_v4_exp),recall_score(y_test, y_pred_v5_exp),recall_score(y_test, y_pred_v6_exp),recall_score(y_test, y_pred_v7_exp),recall_score(y_test, y_pred_v8_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v1_exp,pos_label=0),recall_score(y_test, y_pred_v2_exp,pos_label=0),recall_score(y_test, y_pred_v3_exp,pos_label=0),recall_score(y_test, y_pred_v4_exp,pos_label=0),recall_score(y_test, y_pred_v5_exp,pos_label=0),recall_score(y_test, y_pred_v6_exp,pos_label=0),recall_score(y_test, y_pred_v7_exp,pos_label=0),recall_score(y_test, y_pred_v8_exp,pos_label=0)],
                 'f1_test':[f1_score(y_test, y_pred_v1_exp), f1_score(y_test, y_pred_v2_exp), f1_score(y_test, y_pred_v3_exp),f1_score(y_test, y_pred_v4_exp),f1_score(y_test, y_pred_v5_exp),f1_score(y_test, y_pred_v6_exp), f1_score(y_test, y_pred_v7_exp), f1_score(y_test, y_pred_v8_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v1_exp),precision_score(y_test, y_pred_v2_exp),precision_score(y_test, y_pred_v3_exp),precision_score(y_test, y_pred_v4_exp),precision_score(y_test, y_pred_v5_exp),precision_score(y_test, y_pred_v6_exp),precision_score(y_test, y_pred_v7_exp),precision_score(y_test, y_pred_v8_exp)],                                 
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v1_exp),roc_auc_score(y_test, y_pred_v2_exp),roc_auc_score(y_test, y_pred_v3_exp),roc_auc_score(y_test, y_pred_v4_exp),roc_auc_score(y_test, y_pred_v4_exp),roc_auc_score(y_test, y_pred_v6_exp),roc_auc_score(y_test, y_pred_v7_exp),roc_auc_score(y_test, y_pred_v8_exp)]    
    }


df_overall_results_paper=pd.DataFrame(data=overall_results)
df_overall_results_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_IEEE\df_test_overall_results_paper.xlsx',index=False)


#decision tree
print('confusion matrix of best estimator',confusion_matrix(y_test, y_pred_v1_exp))
print('confusion matrix of decisiontree',confusion_matrix(y_test, y_pred_v2_exp))
print('confusion matrix of rndforest',confusion_matrix(y_test, y_pred_v3_exp))
print('confusion matrix of extratree',confusion_matrix(y_test, y_pred_v4_exp))
print('confusion matrix of adaboost',confusion_matrix(y_test, y_pred_v5_exp))
print('confusion matrix of graboost',confusion_matrix(y_test, y_pred_v6_exp))
print('confusion matrix of xgboost',confusion_matrix(y_test, y_pred_v7_exp))
print('confusion matrix of maxVoting',confusion_matrix(y_test, y_pred_v8_exp))



########### Disclosing feature selected  

#Building pipeline's branch per separate to extract the feature 
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

# Depending on the estimator
###clf v63_exp: ExtraTrees, median, numRFE-4,nomChiSquared-4 
feature_select.feat_sel_RFE(df_X_train_numfeatsel,y_train,k_out_features=4)
#numerical_features_v63_exp=['sc','pot','hemo','rc']
feature_select.feat_sel_Cat_to_Cat_chi2(df_X_train_catfeatsel,y_train,k_out_features=4)
# category_features_v63_exp= ['al','su','htn', 'dm']


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
