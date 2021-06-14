#Created on Fri Apr 9th 2021

#%%

#Import general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import libraries useful for building the pipeline and join their branches
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


#import modules created for data preparation phase
import my_utils
import missing_val_imput
import feature_select
import preprocessing
import adhoc_transf

#import libraries for data preparation phase
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder


#import libraries from modelling phase
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

#import classifiers
#import Ensemble Trees Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
import xgboost as xgb

#to save model fit with GridSearchCV and avoid longer waits
import joblib

#%%

#Loading the dataset
#path_data: XXX
path_data=r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\kidney_disease.csv'

df=pd.read_csv(path_data)
df.head()

#%%Characterizing the data set
target_feature='classification'
numerical_feats=['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
nominal_feats=['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
ordinal_feats=['sg','al','su',]

len_numerical_feats=len(numerical_feats)
len_nominal_feats=len(nominal_feats)
len_ordinal_feats=len(ordinal_feats)

######################################
#Step 0: Perform EDA to detect missing values, imbalanced data, strange characters,etc.
#############################
#%%
##Statistical analysis
df.describe()
#%%
#Identifying missing values
my_utils.info_adhoc(df)
#%%
#Exploring wrong characters
my_utils.df_values(df)

#%%
#Adhoc functions to manage wrong characters
#Below this an example for CKD dataset
def misspellingCorrector(df):
    df.iloc[:] = df.iloc[:].str.replace(r'\t','')
    df.iloc[:] = df.iloc[:].str.replace(r' ','')
    return df


#%%
#############################
#Step 1 Solving wrong characters of dataset
#############################
#Set column id as index


# CKD case does only have misspellingCorrector
# df_content_solver=Pipeline([('fx1', misspellingCorrector()),
#                             ('fx2',function2()),
#                             ('fx3',function3())
# ])

#%%
df.set_index('id', inplace=True)
#%%
feat_list =['classification','dm','cad']
#df.loc[:,feat_list]=adhoc_transf.misspellingTransformer().fit_transform(df.loc[:,feat_list])
for i in feat_list:
     print('i',i)
     df.loc[:,i]=misspellingCorrector((df.loc[:,i]))
my_utils.df_values(df)

#%%Performing numeric cast for numerical features
df.loc[:,numerical_feats]=adhoc_transf.Numeric_Cast_Column().fit_transform(df.loc[:,numerical_feats])
df[numerical_feats].dtypes


#%%Performing category cast for nominal features
df.loc[:,nominal_feats]=adhoc_transf.Category_Cast_Column().fit_transform(df.loc[:,nominal_feats])
df[nominal_feats].dtypes

#%%Performing category cast for ordinal features
df.loc[:,ordinal_feats]=adhoc_transf.Category_Cast_Column().fit_transform(df.loc[:,ordinal_feats])
df[ordinal_feats].dtypes


#%% downcasting last category value of features 'al' and 'su' from 5 to 4
feat_list_tocast=['al','su']
df.loc[:,feat_list_tocast]=adhoc_transf.CastDown().fit_transform(df.loc[:,feat_list_tocast])

#%%
#Exploring wrong characters
my_utils.df_values(df)

#%%
#############################
##Step 2 Train-Test splitting
#############################

#Split the dataset into train and test
test_ratio_split=0.3
train_set,test_set=train_test_split(df, test_size=test_ratio_split, random_state=42, stratify=df[target_feature])

X_train=train_set.drop(target_feature,axis=1)
y_train=train_set[target_feature].copy()

X_test=test_set.drop(target_feature,axis=1)
y_test=test_set[target_feature].copy()

#%%
########################################
##Step 3 Label Encoding of target value
########################################
le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)
le.classes_
#%%
##############################
##Step 2 Building pipelines for data preparation
##############################

#Lets define 3 pipeline mode
#a) parallel approach where feature selection is performed in parallel 
# for numerical, nominal and categorical
#b) general approach where feature selection is performed as a whole for other features
#c) no feature selection is performed

#Before a data preprocessing will take place for each type of feature

pipeline_numeric_feat=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median')),
                                 ('scaler', MinMaxScaler())])

pipeline_numeric_feat_mean=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='mean')),
                                 ('scaler', MinMaxScaler())])

pipeline_nominal_feat=Pipeline([('data_missing',missing_val_imput.Category_Imputer()),                                 
                                 ('encoding', OrdinalEncoder())])#We dont use OneHotEncoder since it enlarges the number of nominal features 

pipeline_ordinal_feat=Pipeline([ ('data_missing',missing_val_imput.Category_Imputer(strategy='most_frequent')),
                                 ('encoding', OrdinalEncoder())])


#option a)
pipe_numeric_featsel=Pipeline([('data_prep',pipeline_numeric_feat),
                                ('feat_sel',feature_select.Feature_Selector(strategy='wrapper_RFECV') )])
pipe_nominal_featsel=Pipeline([('data_prep',pipeline_nominal_feat),
                                ('feat_sel',feature_select.Feature_Selector(strategy='wrapper_RFECV') )])
pipe_ordinal_featsel=Pipeline([('data_prep',pipeline_ordinal_feat),
                                ('feat_sel',feature_select.Feature_Selector(strategy='wrapper_RFECV') )])

dataprep_pipe_opta=ColumnTransformer([('numeric_pipe',pipe_numeric_featsel,numerical_feats),
                                    ('nominal_pipe',pipe_nominal_featsel,nominal_feats),
                                    ('ordinal_pipe',pipe_ordinal_featsel,ordinal_feats)
                                ])

#option b)
dataprep_merge_feat=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_feats),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats),
                                    ('ordinal_pipe',pipeline_ordinal_feat,ordinal_feats)
                                ])
dataprep_pipe_optb=Pipeline([('data_prep',dataprep_merge_feat),
                                ('feat_sel',feature_select.Feature_Selector(strategy='wrapper_RFECV') )])

#option c)
#dataprep_merge_feat is used
dataprep_merge_feat
#%%
#pipe_nominal_featsel.fit_transform(X_train[nominal_feats],y_train)
#df_test=pipe_nominal_featsel.transform(df[nominal_feats],y_train)
#%%
#############################
##Step 3 Classifier initialization
#############################
#Several ensemble classifier with Cross validation will be applied
#we take decision tree as base classifier

#Init the clasfifier
dectree_clf=DecisionTreeClassifier(random_state=42)
rndforest_clf=RandomForestClassifier(random_state=42)
extratree_clf=ExtraTreesClassifier(random_state=42)
ada_clf= AdaBoostClassifier(random_state=42)
xgboost_clf= xgb.XGBClassifier(random_state=42)
gradboost_clf=GradientBoostingClassifier(random_state=42)
voting_clf=VotingClassifier(estimators=[('rdf', rndforest_clf), ('xtra', extratree_clf), ('ada', ada_clf)], voting='soft')
#

#%%
#############################
##Step 4 Scoring initialization
#############################

#Lets define the scoring for the GridSearchCV
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0),
    'precision':make_scorer(precision_score),
    'f1':make_scorer(f1_score),
    'roc_auc':make_scorer(roc_auc_score),
    'mcc':make_scorer(matthews_corrcoef)    
}

#%%
#################################################
##Step 5 Training the data set with GridSearchCV
#################################################


##5.a Parallel approach
#######################
full_parallel_pipe_opta=Pipeline([('data_prep',dataprep_pipe_opta),('clf',dectree_clf)])

full_parallel_pipe_opta.get_params().keys()



#%% Load the model saved to avoid a new fitting
clf_fpipe_a= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_fpipe_a.pkl')

#%%
param_grid_fpipe_a={'clf':[dectree_clf, rndforest_clf, extratree_clf, ada_clf, xgboost_clf,gradboost_clf,voting_clf ],
                    'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean','median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[*range(1,len_numerical_feats+1)],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num','filter_mutinf','wrapper_RFE'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[*range(1,len_nominal_feats+1)],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE'],
                    'data_prep__ordinal_pipe__feat_sel__k_out_features':[*range(1,len_ordinal_feats+1)],
                    'data_prep__ordinal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE']
                    }

# param_grid_fpipe_a={'clf':[dectree_clf, rndforest_clf],
#                     'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
#                      'data_prep__numeric_pipe__feat_sel__k_out_features':[1,2,3],
#                      'data_prep__numeric_pipe__feat_sel__strategy':['filter_num'],
#                      'data_prep__nominal_pipe__feat_sel__k_out_features':[1,2,3],
#                      'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat'],
#                      'data_prep__ordinal_pipe__feat_sel__k_out_features':[1,2,3],
#                      'data_prep__ordinal_pipe__feat_sel__strategy':['filter_mutinf']
#                     }

clf_fpipe_a=GridSearchCV(full_parallel_pipe_opta,param_grid_fpipe_a,scoring=scoring,refit='accuracy', cv=5,n_jobs=-1)
clf_fpipe_a.fit(X_train,y_train)

#%% Saving the model
joblib.dump(clf_fpipe_a, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_fpipe_a.pkl', compress=1)

#
#%% Printing the best estimator 
print('Best estimator of clf_fpipe_a:', clf_fpipe_a.best_params_)
print('Params of best estimator of clf_fpipe_a:', clf_fpipe_a.best_params_)
print('Score of best estimator of clf_fpipe_a:', clf_fpipe_a.best_score_)

#%%
print('Best index',clf_fpipe_a.best_index_ )

#%% Saving the training results into dataframe
df_results_clf_fpipe_a=pd.DataFrame(clf_fpipe_a.cv_results_)
# create an excel with the cross val resutls
df_results_clf_fpipe_a.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\train_results_clf_fpipe_a.xlsx',index=True)

#%% Performing test phase with test set 
clf_fpipe_a.refit
y_pred_clf_fpipe_a=clf_fpipe_a.predict(X_test)
test_results_clf_fpipe_a={'clf':['clf_fpipe_a'],
                 'params':[clf_fpipe_a.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_a)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_a)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_a)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_a)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_a,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_a)]    
    }
#%%
test_results_y_pred_clf_fpipe_a=pd.DataFrame(data=test_results_clf_fpipe_a)
test_results_y_pred_clf_fpipe_a.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\test_results_y_pred_clf_fpipe_a.xlsx',index=False)
print('Accuracy of test set',accuracy_score(y_test, y_pred_clf_fpipe_a))

#%%
##5.b general approach where feature selection is performed as a whole for other features
#########################################################################################
full_parallel_pipe_optb=Pipeline([('data_prep',dataprep_pipe_optb),('clf',dectree_clf)])

full_parallel_pipe_optb.get_params().keys()

#%% Load the model saved to avoid a new fitting
clf_fpipe_b= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_fpipe_b.pkl')

#%%
param_grid_fpipe_b={'clf':[dectree_clf, rndforest_clf, extratree_clf, ada_clf, xgboost_clf,gradboost_clf,voting_clf ],
                    'data_prep__data_prep__numeric_pipe__data_missing__strategy':['mean','median'],
                    'data_prep__feat_sel__k_out_features':[*range(1,len_numerical_feats+len_nominal_feats+len_ordinal_feats+1)],
                    'data_prep__feat_sel__strategy':['filter_mutinf','wrapper_RFE']
                    }

# param_grid_fpipe_b={'clf':[dectree_clf, rndforest_clf ],
#                     'data_prep__data_prep__numeric_pipe__data_missing__strategy':['mean','median'],
#                     'data_prep__feat_sel__k_out_features':[1,2,3],
#                     'data_prep__feat_sel__strategy':['filter_mutinf']
#                     }

clf_fpipe_b=GridSearchCV(full_parallel_pipe_optb,param_grid_fpipe_b,scoring=scoring,refit='accuracy', cv=5,n_jobs=-1)
clf_fpipe_b.fit(X_train,y_train)

#%% Saving the model
joblib.dump(clf_fpipe_b, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_fpipe_b.pkl', compress=1)

#
#%% Printing the best estimator 
print('Best estimator of clf_fpipe_b:', clf_fpipe_b.best_params_)
print('Params of best estimator of clf_fpipe_b:', clf_fpipe_b.best_params_)
print('Score of best estimator of clf_fpipe_b:', clf_fpipe_b.best_score_)

#%% Saving the training results into dataframe
df_results_clf_fpipe_b=pd.DataFrame(clf_fpipe_b.cv_results_)
# create an excel with the cross val resutls
df_results_clf_fpipe_b.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\train_results_clf_fpipe_b.xlsx',index=True)

#%% Performing test phase with test set 
clf_fpipe_b.refit
y_pred_clf_fpipe_b=clf_fpipe_b.predict(X_test)
test_results_clf_fpipe_b={'clf':['clf_fpipe_b'],
                 'params':[clf_fpipe_b.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_b)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_b)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_b)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_b)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_b,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_b)]    
    }
#%%
test_results_y_pred_clf_fpipe_b=pd.DataFrame(data=test_results_clf_fpipe_b)
test_results_y_pred_clf_fpipe_b.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\test_results_y_pred_clf_fpipe_b.xlsx',index=False)
print('Accuracy of test set',accuracy_score(y_test, y_pred_clf_fpipe_b))

#%%
##5.c general approach where feature selection is performed as a whole for other features
#########################################################################################
full_parallel_pipe_optc=Pipeline([('data_prep',dataprep_merge_feat),('clf',dectree_clf)])

full_parallel_pipe_optc.get_params().keys()
# %%
#%% Load the model saved to avoid a new fitting
clf_fpipe_c= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_fpipe_c.pkl')

#%%
param_grid_fpipe_c={'clf':[dectree_clf, rndforest_clf, extratree_clf, ada_clf, xgboost_clf,gradboost_clf,voting_clf ],
                    'data_prep__numeric_pipe__data_missing__strategy':['mean','median']
                    }

# param_grid_fpipe_c={'clf':[dectree_clf, rndforest_clf ],
#                     'data_prep__numeric_pipe__data_missing__strategy':['mean','median']
#                     }

clf_fpipe_c=GridSearchCV(full_parallel_pipe_optc,param_grid_fpipe_c,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_fpipe_c.fit(X_train,y_train)

# %%#%% Saving the model
joblib.dump(clf_fpipe_c, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_fpipe_c.pkl', compress=1)

#
#%% Printing the best estimator 
print('Best estimator of clf_fpipe_c:', clf_fpipe_c.best_params_)
print('Params of best estimator of clf_fpipe_c:', clf_fpipe_c.best_params_)
print('Score of best estimator of clf_fpipe_a:', clf_fpipe_c.best_score_)


#%% Saving the training results into dataframe
df_results_clf_fpipe_c=pd.DataFrame(clf_fpipe_c.cv_results_)
# create an excel with the cross val resutls
df_results_clf_fpipe_c.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\train_results_clf_fpipe_c.xlsx',index=True)

#%% Performing test phase with test set 
clf_fpipe_c.refit
y_pred_clf_fpipe_c=clf_fpipe_c.predict(X_test)
test_results_clf_fpipe_c={'clf':['clf_fpipe_c'],
                 'params':[clf_fpipe_c.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_c)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_c)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_c)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_c)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_c,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_c)]    
    }

#%%
test_results_y_pred_clf_fpipe_c=pd.DataFrame(data=test_results_clf_fpipe_c)
test_results_y_pred_clf_fpipe_c.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\test_results_y_pred_clf_fpipe_c.xlsx',index=False)
print('Accuracy of test set',accuracy_score(y_test, y_pred_clf_fpipe_c))

#######################
#Build a param grid for the best combination per each estimator considered
#v2:DecisionTree
#######################
#%%
param_grid_v2_exp={'clf': [dectree_clf],
                    'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[1],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[2],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat'],
                    'data_prep__ordinal_pipe__feat_sel__k_out_features':[3],
                    'data_prep__ordinal_pipe__feat_sel__strategy':['filter_cat']
     }

clf_v2_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v2_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v2_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v2_exp:', clf_v2_exp.best_score_) #Score of best estimator of clf_v2: 0.9964285714285716

#%%
#Saving the results in an excel
df_results_v2_exp=pd.DataFrame(clf_v2_exp.cv_results_)
df_results_v2_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\df_results_v2_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v2_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_v2_exp.pkl', compress=1)

#Obtaining classification  with test set
clf_v2_exp.refit
y_pred_v2_exp = clf_v2_exp.predict(X_test)

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
test_results_DT_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\test_results_DT_paper.xlsx',index=False)

#%%
######################
#v3:Random Forest
#######################

param_grid_v3_exp={'clf': [rndforest_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[1],
                    'data_prep__numeric_pipe__feat_sel__strategy':['wrapper_RFE'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[3],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_mutinf'],
                    'data_prep__ordinal_pipe__feat_sel__k_out_features':[2],
                    'data_prep__ordinal_pipe__feat_sel__strategy':['filter_mutinf']
     }

clf_v3_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v3_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v3_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v3_exp:', clf_v3_exp.best_score_) #Score of best estimator of clf_v3: 1

#Saving the results in an excel
df_results_v3_exp=pd.DataFrame(clf_v3_exp.cv_results_)
df_results_v3_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\df_results_v3_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v3_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_v3_exp.pkl', compress=1)

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
test_results_RF_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\test_results_RF_paper.xlsx',index=False)


#%%
######################
#v4:Extra Trees
#######################

param_grid_v4_exp={'clf': [extratree_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[8],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[2],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat'],
                    'data_prep__ordinal_pipe__feat_sel__k_out_features':[2],
                    'data_prep__ordinal_pipe__feat_sel__strategy':['filter_mutinf']
     }

clf_v4_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v4_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v4_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v4_exp:', clf_v4_exp.best_score_) #Score of best estimator of clf_v4: 1

#Saving the results in an excel
df_results_v4_exp=pd.DataFrame(clf_v4_exp.cv_results_)
df_results_v4_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\df_results_v4_exp.xlsx',index=False)
#Saving the model
#joblib.dump(clf_v4_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_v4_exp.pkl', compress=1)

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
test_results_ET_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\test_results_ET_paper.xlsx',index=False)

#%%
######################
#v5:AdaBoost
#######################

param_grid_v5_exp={'clf': [ada_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[4],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[4],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat'],
                    'data_prep__ordinal_pipe__feat_sel__k_out_features':[2],
                    'data_prep__ordinal_pipe__feat_sel__strategy':['filter_mutinf']
     }

clf_v5_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v5_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v5_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v5_exp:', clf_v5_exp.best_score_) #Score of best estimator of clf_v5: 1

#Saving the results in an excel
df_results_v5_exp=pd.DataFrame(clf_v5_exp.cv_results_)
df_results_v5_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\df_results_v5_exp.xlsx',index=False)
#Saving the model
#joblib.dump(clf_v5_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_v5_exp.pkl', compress=1)

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
test_results_AB_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\test_results_AB_paper.xlsx',index=False)

#%%

######################
#v6:Gradient Boosting
#######################

param_grid_v6_exp={'clf': [gradboost_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[1],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[4],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat'],
                    'data_prep__ordinal_pipe__feat_sel__k_out_features':[1],
                    'data_prep__ordinal_pipe__feat_sel__strategy':['filter_mutinf']
     }

clf_v6_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v6_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v6_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v6_exp:', clf_v6_exp.best_score_) #Score of best estimator of clf_v6: 0.9928571428571429

#Saving the results in an excel
df_results_v6_exp=pd.DataFrame(clf_v6_exp.cv_results_)
df_results_v6_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\df_results_v6_exp.xlsx',index=False)
#Saving the model
#joblib.dump(clf_v6_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_v6_exp.pkl', compress=1)

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
test_results_GB_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\test_results_GB_paper.xlsx',index=False)

#%%
######################
#v7:eXtreme Gradient Boosting
#######################

param_grid_v7_exp={'clf': [xgboost_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[1],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_mutinf'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[1],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat'],
                    'data_prep__ordinal_pipe__feat_sel__k_out_features':[2],
                    'data_prep__ordinal_pipe__feat_sel__strategy':['wrapper_RFE']
     }

clf_v7_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v7_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v7_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v7_exp:', clf_v7_exp.best_score_) #Score of best estimator of clf_v7: 0.9857142857142858

#Saving the results in an excel
df_results_v7_exp=pd.DataFrame(clf_v7_exp.cv_results_)
df_results_v7_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\df_results_v7_exp.xlsx',index=False)
#Saving the model
#joblib.dump(clf_v7_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_v7_exp.pkl', compress=1)

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
test_results_XGB_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\test_results_XGB_paper.xlsx',index=False)

#%%
######################
#v8:Max Voting
#######################

param_grid_v8_exp={'clf': [voting_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[1],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_mutinf'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[4],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat'],
                    'data_prep__ordinal_pipe__feat_sel__k_out_features':[3],
                    'data_prep__ordinal_pipe__feat_sel__strategy':['filter_cat',]
     }

clf_v8_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v8_exp,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_v8_exp.fit(X_train,y_train)

print('Score of best estimator of clf_v8_exp:', clf_v8_exp.best_score_) #Score of best estimator of clf_v8: 1

#Saving the results in an excel
df_results_v8_exp=pd.DataFrame(clf_v8_exp.cv_results_)
df_results_v8_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\df_results_v8_exp.xlsx',index=False)
#Saving the model
#joblib.dump(clf_v8_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\clf_v8_exp.pkl', compress=1)

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
test_results_MaxV_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\test_results_MaxV_paper.xlsx',index=False)

#%%
#########################
#Dataframe with the best estimator per each classifier applied to the Xtest
# #######################

overall_results={'clf':['clf_v1_exp','clf_v2_exp','clf_v3_exp','clf_v4_exp','clf_v5_exp','clf_v6_exp','clf_v7_exp','clf_v8_exp'],
                 'params':[clf_fpipe_a.best_params_, clf_v2_exp.best_params_, clf_v3_exp.best_params_, clf_v4_exp.best_params_,clf_v5_exp.best_params_,clf_v6_exp.best_params_,clf_v7_exp.best_params_,clf_v8_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_a),accuracy_score(y_test, y_pred_v2_exp),accuracy_score(y_test, y_pred_v3_exp),accuracy_score(y_test, y_pred_v4_exp),accuracy_score(y_test, y_pred_v5_exp),accuracy_score(y_test, y_pred_v6_exp),accuracy_score(y_test, y_pred_v7_exp),accuracy_score(y_test, y_pred_v8_exp)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_a),recall_score(y_test, y_pred_v2_exp),recall_score(y_test, y_pred_v3_exp),recall_score(y_test, y_pred_v4_exp),recall_score(y_test, y_pred_v5_exp),recall_score(y_test, y_pred_v6_exp),recall_score(y_test, y_pred_v7_exp),recall_score(y_test, y_pred_v8_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_a,pos_label=0),recall_score(y_test, y_pred_v2_exp,pos_label=0),recall_score(y_test, y_pred_v3_exp,pos_label=0),recall_score(y_test, y_pred_v4_exp,pos_label=0),recall_score(y_test, y_pred_v5_exp,pos_label=0),recall_score(y_test, y_pred_v6_exp,pos_label=0),recall_score(y_test, y_pred_v7_exp,pos_label=0),recall_score(y_test, y_pred_v8_exp,pos_label=0)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_a), f1_score(y_test, y_pred_v2_exp), f1_score(y_test, y_pred_v3_exp),f1_score(y_test, y_pred_v4_exp),f1_score(y_test, y_pred_v5_exp),f1_score(y_test, y_pred_v6_exp), f1_score(y_test, y_pred_v7_exp), f1_score(y_test, y_pred_v8_exp)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_a),precision_score(y_test, y_pred_v2_exp),precision_score(y_test, y_pred_v3_exp),precision_score(y_test, y_pred_v4_exp),precision_score(y_test, y_pred_v5_exp),precision_score(y_test, y_pred_v6_exp),precision_score(y_test, y_pred_v7_exp),precision_score(y_test, y_pred_v8_exp)],                                 
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_a),roc_auc_score(y_test, y_pred_v2_exp),roc_auc_score(y_test, y_pred_v3_exp),roc_auc_score(y_test, y_pred_v4_exp),roc_auc_score(y_test, y_pred_v4_exp),roc_auc_score(y_test, y_pred_v6_exp),roc_auc_score(y_test, y_pred_v7_exp),roc_auc_score(y_test, y_pred_v8_exp)]    
    }


df_overall_results_paper=pd.DataFrame(data=overall_results)
df_overall_results_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\df_test_overall_results_paper.xlsx',index=False)


#decision tree
# print('confusion matrix of best estimator',confusion_matrix(y_test, y_pred_v1_exp))
# print('confusion matrix of decisiontree',confusion_matrix(y_test, y_pred_v2_exp))
# print('confusion matrix of rndforest',confusion_matrix(y_test, y_pred_v3_exp))
# print('confusion matrix of extratree',confusion_matrix(y_test, y_pred_v4_exp))
# print('confusion matrix of adaboost',confusion_matrix(y_test, y_pred_v5_exp))
# print('confusion matrix of graboost',confusion_matrix(y_test, y_pred_v6_exp))
# print('confusion matrix of xgboost',confusion_matrix(y_test, y_pred_v7_exp))
# print('confusion matrix of maxVoting',confusion_matrix(y_test, y_pred_v8_exp))

#%%
#######################################
#Applying feature selection for conduct explainability analysis
#######################################
X_train_feat=X_train.copy()
X_train_feat[numerical_feats]=pipeline_numeric_feat.fit_transform(X_train_feat[numerical_feats])
X_train_feat[nominal_feats]=pipeline_nominal_feat.fit_transform(X_train_feat[nominal_feats])
X_train_feat[ordinal_feats]=pipeline_ordinal_feat.fit_transform(X_train_feat[ordinal_feats])

# %%
#ANOVA for numerical features median imputation
feature_select.feat_sel_Num_to_Cat(X_train_feat[numerical_feats], y_train, 'all')
# Feature of feat_sel_Num_to_Cat age: 14.903648
# Feature of feat_sel_Num_to_Cat bp: 35.199138
# Feature of feat_sel_Num_to_Cat bgr: 43.816386
# Feature of feat_sel_Num_to_Cat bu: 39.848049
# Feature of feat_sel_Num_to_Cat sc: 32.735196
# Feature of feat_sel_Num_to_Cat sod: 59.375789
# Feature of feat_sel_Num_to_Cat pot: 0.979238
# Feature of feat_sel_Num_to_Cat hemo: 302.396682
# Feature of feat_sel_Num_to_Cat pcv: 226.943372
# Feature of feat_sel_Num_to_Cat wc: 6.506129
# Feature of feat_sel_Num_to_Cat rc: 131.422025

# %%
#Multinf for numerical features median imputation
feature_select.feat_sel_Cat_to_Cat_mutinf(X_train_feat[numerical_feats], y_train, 'all')

# Feature of feat_sel_Cat_to_Cat mutual info age: 0.059238
# Feature of feat_sel_Cat_to_Cat mutual info bp: 0.077888
# Feature of feat_sel_Cat_to_Cat mutual info bgr: 0.179110
# Feature of feat_sel_Cat_to_Cat mutual info bu: 0.216736
# Feature of feat_sel_Cat_to_Cat mutual info sc: 0.344144
# Feature of feat_sel_Cat_to_Cat mutual info sod: 0.227409
# Feature of feat_sel_Cat_to_Cat mutual info pot: 0.238933
# Feature of feat_sel_Cat_to_Cat mutual info hemo: 0.443066
# Feature of feat_sel_Cat_to_Cat mutual info pcv: 0.471854
# Feature of feat_sel_Cat_to_Cat mutual info wc: 0.111779
# Feature of feat_sel_Cat_to_Cat mutual info rc: 0.404427

#%%
#ANOVA for numerical features mean imputation
X_train_feat[numerical_feats]=pipeline_numeric_feat_mean.fit_transform(X_train_feat[numerical_feats])
feature_select.feat_sel_Num_to_Cat(X_train_feat[numerical_feats], y_train, 'all')

# Feature of feat_sel_Num_to_Cat age: 14.903648
# Feature of feat_sel_Num_to_Cat bp: 35.199138
# Feature of feat_sel_Num_to_Cat bgr: 43.816386
# Feature of feat_sel_Num_to_Cat bu: 39.848049
# Feature of feat_sel_Num_to_Cat sc: 32.735196
# Feature of feat_sel_Num_to_Cat sod: 59.375789
# Feature of feat_sel_Num_to_Cat pot: 0.979238
# Feature of feat_sel_Num_to_Cat hemo: 302.396682
# Feature of feat_sel_Num_to_Cat pcv: 226.943372
# Feature of feat_sel_Num_to_Cat wc: 6.506129
# Feature of feat_sel_Num_to_Cat rc: 131.422025

# %%
#Multinf for numerical features mean imputation
feature_select.feat_sel_Cat_to_Cat_mutinf(X_train_feat[numerical_feats], y_train, 'all')

# Feature of feat_sel_Cat_to_Cat mutual info age: 0.056812
# Feature of feat_sel_Cat_to_Cat mutual info bp: 0.124319
# Feature of feat_sel_Cat_to_Cat mutual info bgr: 0.193173
# Feature of feat_sel_Cat_to_Cat mutual info bu: 0.203115
# Feature of feat_sel_Cat_to_Cat mutual info sc: 0.379725
# Feature of feat_sel_Cat_to_Cat mutual info sod: 0.249214
# Feature of feat_sel_Cat_to_Cat mutual info pot: 0.262116
# Feature of feat_sel_Cat_to_Cat mutual info hemo: 0.451713
# Feature of feat_sel_Cat_to_Cat mutual info pcv: 0.467550
# Feature of feat_sel_Cat_to_Cat mutual info wc: 0.138030
# Feature of feat_sel_Cat_to_Cat mutual info rc: 0.407542

#%%
#RFE for numerical features
feature_select.feat_sel_RFE(X_train_feat[numerical_feats], y_train, 1)
# Num Features: 1
# Selected Features: [False False False False False False False  True False False False]
# Feature Ranking: [10  5  4  8  7  6 11  1  2  9  3]

# %%
#Chi2 for nominal features
feature_select.feat_sel_Cat_to_Cat_chi2(X_train_feat[nominal_feats], y_train, 'all')
# Feature of feat_sel_Cat_to_Cat chi2 rbc: 3.380247
# Feature of feat_sel_Cat_to_Cat chi2 pc: 7.424670
# Feature of feat_sel_Cat_to_Cat chi2 pcc: 16.800000
# Feature of feat_sel_Cat_to_Cat chi2 ba: 9.600000
# Feature of feat_sel_Cat_to_Cat chi2 htn: 58.200000
# Feature of feat_sel_Cat_to_Cat chi2 dm: 55.800000
# Feature of feat_sel_Cat_to_Cat chi2 cad: 12.600000
# Feature of feat_sel_Cat_to_Cat chi2 appet: 34.800000
# Feature of feat_sel_Cat_to_Cat chi2 pe: 31.800000
# Feature of feat_sel_Cat_to_Cat chi2 ane: 21.000000

#%%
#Multinf for nominal features
feature_select.feat_sel_Cat_to_Cat_mutinf(X_train_feat[nominal_feats], y_train, 'all')

#Multinf for nominal features...
# Feature of feat_sel_Cat_to_Cat mutual info rbc: 0.046007
# Feature of feat_sel_Cat_to_Cat mutual info pc: 0.116813
# Feature of feat_sel_Cat_to_Cat mutual info pcc: 0.066076
# Feature of feat_sel_Cat_to_Cat mutual info ba: 0.001647
# Feature of feat_sel_Cat_to_Cat mutual info htn: 0.173396
# Feature of feat_sel_Cat_to_Cat mutual info dm: 0.217810
# Feature of feat_sel_Cat_to_Cat mutual info cad: 0.047517
# Feature of feat_sel_Cat_to_Cat mutual info appet: 0.143099
# Feature of feat_sel_Cat_to_Cat mutual info pe: 0.080126
# Feature of feat_sel_Cat_to_Cat mutual info ane: 0.047926

# %%
#Chi2 for ordinal features
feature_select.feat_sel_Cat_to_Cat_chi2(X_train_feat[ordinal_feats], y_train, 'all')
# Feature of feat_sel_Cat_to_Cat chi2 sg: 56.134728
# Feature of feat_sel_Cat_to_Cat chi2 al: 153.600000
# Feature of feat_sel_Cat_to_Cat chi2 su: 72.600000

# %%#Multinf for ordinal features
feature_select.feat_sel_Cat_to_Cat_mutinf(X_train_feat[ordinal_feats], y_train, 'all')
# Feature of feat_sel_Cat_to_Cat mutual info sg: 0.311586
# Feature of feat_sel_Cat_to_Cat mutual info al: 0.285203
# Feature of feat_sel_Cat_to_Cat mutual info su: 0.092803

#%%
#RFE for ordinal features
feature_select.feat_sel_RFE(X_train_feat[ordinal_feats], y_train, 2)
# Num Features: 2
# Selected Features: [ True  True]
# Feature Ranking: [1 1]
# sg	al


# %%
#Building the decision tree to calculate the fidelity score for  each classifier's best estimator
# Fidelity formula --> F=acc(decision_tree)/acc(best classifier's estimator )

#Random Forest
pipeline_numeric_feat_median=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median')),
                                 ('scaler', MinMaxScaler())])
numerical_feats_rf=['hemo']
nominal_feats_rf=[ 'htn', 'dm', 'appet']
ordinal_feats_rf=['sg','al']
dataprep_merge_feat_rf=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat_median,numerical_feats_rf),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_rf),
                                    ('ordinal_pipe',pipeline_ordinal_feat,ordinal_feats_rf)
                                    ])

pipe_fidelity_rf=Pipeline([('data_prep',dataprep_merge_feat_rf),
                          ('clf', dectree_clf)])

pipe_fidelity_rf.fit(X_train,y_train)
y_pred_fidelity_rf = pipe_fidelity_rf.predict(X_test)
accuracy_score(y_test, y_pred_fidelity_rf)#0.9916666666666667

# %%
#ExtraTrees
pipeline_numeric_feat_median=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median')),
                                 ('scaler', MinMaxScaler())])
numerical_feats_et=['bp','bgr','bu','sc','sod','hemo','pcv','rc']
nominal_feats_et=['htn', 'dm']
ordinal_feats_et=['sg','al']

dataprep_merge_feat_et=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat_median,numerical_feats_et),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_et),
                                    ('ordinal_pipe',pipeline_ordinal_feat,ordinal_feats_et)
                                    ])

pipe_fidelity_et=Pipeline([('data_prep',dataprep_merge_feat_et),
                          ('clf', dectree_clf)])

pipe_fidelity_et.fit(X_train,y_train)
y_pred_fidelity_et = pipe_fidelity_et.predict(X_test)
accuracy_score(y_test, y_pred_fidelity_et)#0.975

# %%
#AdaBoost
pipeline_numeric_feat_median=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median')),
                                 ('scaler', MinMaxScaler())])
numerical_feats_ab=['sod','hemo','pcv','rc']
nominal_feats_ab=[ 'htn', 'dm', 'appet', 'pe']
ordinal_feats_ab=['sg','al']

dataprep_merge_feat_ab=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat_median,numerical_feats_ab),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_ab),
                                    ('ordinal_pipe',pipeline_ordinal_feat,ordinal_feats_ab)
                                    ])

pipe_fidelity_ab=Pipeline([('data_prep',dataprep_merge_feat_ab),
                          ('clf', dectree_clf)])

pipe_fidelity_ab.fit(X_train,y_train)
y_pred_fidelity_ab = pipe_fidelity_ab.predict(X_test)
accuracy_score(y_test, y_pred_fidelity_ab)#0.975


# %%
#GradientBoosting
pipeline_numeric_feat_mean=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='mean')),
                                 ('scaler', MinMaxScaler())])

numerical_feats_gb=['hemo']
nominal_feats_gb=['htn', 'dm', 'appet', 'pe']
ordinal_feats_gb=['sg']

dataprep_merge_feat_gb=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat_mean,numerical_feats_gb),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_gb),
                                    ('ordinal_pipe',pipeline_ordinal_feat,ordinal_feats_gb)
                                    ])

pipe_fidelity_gb=Pipeline([('data_prep',dataprep_merge_feat_gb),
                          ('clf', dectree_clf)])

pipe_fidelity_gb.fit(X_train,y_train)
y_pred_fidelity_gb = pipe_fidelity_gb.predict(X_test)
accuracy_score(y_test, y_pred_fidelity_gb)#1.0

# numerical_feats=['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
# nominal_feats=['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
# ordinal_feats=['sg','al','su',]


# %%
#XGBoost
pipeline_numeric_feat_mean=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='mean')),
                                 ('scaler', MinMaxScaler())])

numerical_feats_xgb=['pcv']
nominal_feats_xgb=[ 'htn']
ordinal_feats_xgb=['sg','al']

dataprep_merge_feat_xgb=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat_mean,numerical_feats_xgb),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_xgb),
                                    ('ordinal_pipe',pipeline_ordinal_feat,ordinal_feats_xgb)
                                    ])

pipe_fidelity_xgb=Pipeline([('data_prep',dataprep_merge_feat_xgb),
                          ('clf', dectree_clf)])

pipe_fidelity_xgb.fit(X_train,y_train)
y_pred_fidelity_xgb = pipe_fidelity_xgb.predict(X_test)
accuracy_score(y_test, y_pred_fidelity_xgb)#0.975

# %%
#Voting Classifier
pipeline_numeric_feat_median=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median')),
                                 ('scaler', MinMaxScaler())])

numerical_feats_vc=['pcv']
nominal_feats_vc=['htn', 'dm', 'appet', 'pe']
ordinal_feats_vc=['sg','al','su',]

dataprep_merge_feat_vc=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat_median,numerical_feats_vc),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_vc),
                                    ('ordinal_pipe',pipeline_ordinal_feat,ordinal_feats_vc)
                                    ])

pipe_fidelity_vc=Pipeline([('data_prep',dataprep_merge_feat_vc),
                          ('clf', dectree_clf)])

pipe_fidelity_vc.fit(X_train,y_train)
y_pred_fidelity_vc = pipe_fidelity_vc.predict(X_test)
accuracy_score(y_test, y_pred_fidelity_vc)#0.975


###########################################################################
#  Explainability Analisys
# The "most explainable" classifier is XGBoost by assessing the FIR ratio
# Different explainability method are considered: implicit feature importance, feature permutation, SHAP and PDP
###########################################################################
#%%
features_selected_xgb=['pcv','htn','sg','al']
X_train_feat_sel=X_train[features_selected_xgb]
X_test_feat_sel=X_test[features_selected_xgb]
#%%
#a) The estimator is refited with those feature selected
#########################################################

pipeline_numeric_feat_mean=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='mean')),
                                 ('scaler', MinMaxScaler())])

numerical_feats_xgb=['pcv']
nominal_feats_xgb=[ 'htn']
ordinal_feats_xgb=['sg','al']

dataprep_merge_feat_xgb=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat_mean,numerical_feats_xgb),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_xgb),
                                    ('ordinal_pipe',pipeline_ordinal_feat,ordinal_feats_xgb)
                                    ])



#%%
X_train_featsel=dataprep_merge_feat_xgb.fit_transform(X_train_feat_sel)
df_X_train_featsel=pd.DataFrame(X_train_featsel, columns=features_selected_xgb)
df_X_train_featsel.head()

X_test_featsel=dataprep_merge_feat_xgb.fit_transform(X_test_feat_sel)
df_X_test_featsel=pd.DataFrame(X_test_featsel, columns=features_selected_xgb)
df_X_test_featsel.head()

xgboost_clf.fit(df_X_train_featsel,y_train)


#%%
#b)Implicit feature importance
#####################################################

importances = xgboost_clf.feature_importances_
indices = np.argsort(importances)

plt.title('Implicit Features Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features_selected_xgb[i] for i in indices])
plt.xlabel('Gini Importance')
plt.show()

#%%
import eli5
from eli5 import show_weights
eli5.explain_weights(xgboost_clf, feature_names=features_selected_xgb)

 
# %%
#c) Implicit feature importance for local explainability
###############################################################

y_pred = xgboost_clf.predict(df_X_test_featsel)
print('y_pred',y_pred)
print('y_test',y_test)

#%%
#predicting true negative - the patient DOES HAVE CKD
index_TN = 12
print(df_X_test_featsel.iloc[index_TN])
print('Actual Label:', y_test[index_TN])
print('Predicted Label:', y_pred[index_TN])
eli5.explain_prediction(xgboost_clf,df_X_test_featsel.iloc[index_TN], feature_names=features_selected_xgb)


#%%
#predicting true negative - the patient DOES NOT HAVE CKD
index_TP = 11
print(df_X_test_featsel.iloc[index_TP])
print('Actual Label:', y_test[index_TP])
print('Predicted Label:', y_pred[index_TP])
eli5.explain_prediction(xgboost_clf,df_X_test_featsel.iloc[index_TP], feature_names=features_selected_xgb)

# %%
#d) Feature permutation importance
#############################################################
# With X_train
from eli5.sklearn import PermutationImportance
perm=PermutationImportance(xgboost_clf).fit(df_X_train_featsel,y_train)
feat_perm_df=eli5.explain_weights_df(perm,feature_names=features_selected_xgb)
feat_perm_df

#%%
fig, ax = plt.subplots()
fig.set_size_inches(13.7, 10.27)
ax.bar(feat_perm_df['feature'], feat_perm_df['weight'], yerr=feat_perm_df['std'], align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Importance')
ax.set_xticks(feat_perm_df['feature'])
ax.set_xticklabels(feat_perm_df['feature'])
ax.set_title('Feature Permutation importance distribution ')
ax.yaxis.grid(True)

#%%
# With X_test
from eli5.sklearn import PermutationImportance
perm=PermutationImportance(xgboost_clf).fit(df_X_test_featsel,y_test)
feat_perm_df=eli5.explain_weights_df(perm,feature_names=features_selected_xgb)
feat_perm_df

#%%
fig, ax = plt.subplots()
fig.set_size_inches(13.7, 10.27)
ax.bar(feat_perm_df['feature'], feat_perm_df['weight'], yerr=feat_perm_df['std'], align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Importance')
ax.set_xticks(feat_perm_df['feature'])
ax.set_xticklabels(feat_perm_df['feature'])
ax.set_title('Feature Permutation importance distribution')
ax.yaxis.grid(True)

#%%
# feat_perm_df_pivot=feat_perm_df.pivot_table(columns='feature', values=['weight','std'])   
# feat_perm_array_iteration={'sg':feat_perm_df_pivot['sg'],
#                               'htn':feat_perm_df_pivot['htn'],
#                               'pcv':feat_perm_df_pivot['pcv'],
#                               'al':feat_perm_df_pivot['al']}
# df_feat_perm_array_iteration= pd.DataFrame(data=feat_perm_array_iteration)
# df_feat_perm_array_iteration

# %%
#d) PDP plots
##########################################################

#%%
#import packages
from pdpbox import pdp, get_dataset, info_plots


# %%
pipeline_numeric_imputer_mean=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='mean'))])

pipeline_nominal_imputer=Pipeline([('data_missing',missing_val_imput.Category_Imputer()),
                                 ('encoding', OrdinalEncoder())])#We dont use OneHotEncoder since it enlarges the number of nominal features 

pipeline_ordinal_imputer=Pipeline([ ('data_missing',missing_val_imput.Category_Imputer(strategy='most_frequent'))])

dataimputer_pipe=ColumnTransformer([('numeric_pipe',pipeline_numeric_imputer_mean,numerical_feats_xgb),
                                    ('nominal_pipe',pipeline_nominal_imputer,nominal_feats_xgb),
                                    ('ordinal_pipe',pipeline_ordinal_imputer,ordinal_feats_xgb)
                                ])

X_train_imputed=dataimputer_pipe.fit_transform(X_train_feat_sel)
df_X_train_imputed=pd.DataFrame(X_train_imputed, columns=features_selected_xgb)
df_X_train_imputed.head()

# %%
pipe_pdp_xgb=Pipeline([('data_prep',dataprep_merge_feat_xgb),
                          ('clf', dectree_clf)])

model=pipe_pdp_xgb.fit(df_X_train_imputed,y_train)

pdp_age_Xtrain= pdp.pdp_isolate(model=model, dataset=df_X_train_imputed, model_features=df_X_train_imputed.columns, feature='pcv')
fig,axes=pdp.pdp_plot(pdp_age_Xtrain, 'pcv',plot_pts_dist=True, frac_to_plot=0.5)
axes['pdp_ax']['_pdp_ax'].set_ylim(ymin=0, ymax=2)
plt.show()
# %%
pdp_age_Xtrain= pdp.pdp_isolate(model=model, dataset=df_X_train_imputed, model_features=df_X_train_imputed.columns, feature='htn')
fig,axes=pdp.pdp_plot(pdp_age_Xtrain, 'htn',plot_pts_dist=True, frac_to_plot=0.5)
axes['pdp_ax']['_pdp_ax'].set_ylim(ymin=0, ymax=2)
plt.show()

#%%
pdp_age_Xtrain= pdp.pdp_isolate(model=model, dataset=df_X_train_imputed, model_features=df_X_train_imputed.columns, feature='al')
fig,axes=pdp.pdp_plot(pdp_age_Xtrain, 'al',plot_pts_dist=True, frac_to_plot=0.5)
axes['pdp_ax']['_pdp_ax'].set_ylim(ymin=0, ymax=2)
plt.show()

#%%
pdp_age_Xtrain= pdp.pdp_isolate(model=model, dataset=df_X_train_imputed, model_features=df_X_train_imputed.columns, feature='sg')
fig,axes=pdp.pdp_plot(pdp_age_Xtrain, 'sg',plot_pts_dist=True, frac_to_plot=0.5)
axes['pdp_ax']['_pdp_ax'].set_ylim(ymin=0, ymax=2)
plt.show()

# %%
#e) SHAP explainability - global explainability
#####################################################

import shap
shap.initjs()

pipe_shap_xgb=Pipeline([('data_prep',dataprep_merge_feat_xgb),
                          ('clf', dectree_clf)])

explainer=shap.explainers.Tree(pipe_shap_xgb.named_steps['clf'], pipe_shap_xgb.named_steps['data_prep'].fit_transform(df_X_train_imputed))
shap_values=explainer.shap_values(pipe_shap_xgb.named_steps['data_prep'].fit_transform(df_X_train_imputed))

# %%
shap.summary_plot(shap_values, df_X_train_featsel,plot_type="bar")
# %%
shap.summary_plot(shap_values[0], df_X_train_featsel,plot_type="dot")
# %%
shap.summary_plot(shap_values[1], df_X_train_featsel,plot_type="dot")
# %%
shap.summary_plot(shap_values[0], df_X_train_featsel,plot_type="bar")
# %%
shap.summary_plot(shap_values[1], df_X_train_featsel,plot_type="bar")

# %%
#f) SHAP explainability - local explainability
##########################################################

pipe_fidelity_xgb.fit(X_train_feat_sel,y_train)
y_pred_train= pipe_fidelity_xgb.predict(X_train_feat_sel)

#%%
print('y_train',y_train)
print('y_pred_train',y_pred_train)

#%%
#True negative instance
index_TN_shap=0
print(df_X_train_imputed.iloc[index_TN_shap])
print('Actual Label:', y_train[index_TN_shap])
print('Predicted Label:', y_pred_train[index_TN_shap])
choosen_instance_tn=df_X_train_imputed.loc[[index_TN_shap]]
shap_values_tn = explainer.shap_values(choosen_instance_tn)

# %%
shap_values_tn = explainer.shap_values(choosen_instance_tn)
shap.force_plot(explainer.expected_value[0], shap_values_tn[0], choosen_instance_tn)

# %%
shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values_tn[0], choosen_instance_tn)
#shap.force_plot(explainer.expected value, shapshap_values[index,:], X_train_feat_selected.iloc[index,:])
                       
#%%0
# compute SHAP values

shap_values=explainer(pipe_shap_xgb.named_steps['data_prep'].fit_transform(df_X_train_imputed))

#%%
shap.waterfall_plot(shap_values[0], df_X_train_imputed[0]), 

#%%
#True negative instance
index_TP_shap=1
print(df_X_train_imputed.iloc[index_TP_shap])
print('Actual Label:', y_train[index_TP_shap])
print('Predicted Label:', y_pred_train[index_TP_shap])
choosen_instance_tp=df_X_train_imputed.loc[[index_TP_shap]]
shap_values_tp = explainer.shap_values(choosen_instance_tp)

# %%
shap_values_tp = explainer.shap_values(choosen_instance_tp)
shap.force_plot(explainer.expected_value[0], shap_values_tp[0], choosen_instance_tp)
# %%
# %%
shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values_tn[1], choosen_instance_tn)
#shap.force_plot(explainer.expected value, shapshap_values[index,:], X_train_feat_selected.iloc[index,:])
#%%
explainer.expected_value
# %%
shap.waterfall_plot