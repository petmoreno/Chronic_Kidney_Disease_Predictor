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
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import statsmodels.api as sm


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR 
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

################
##Filter methods
################
#Function to take a df with num attributes and cat target and return df with k_out_features

def feat_sel_Num_to_Cat(X, y, k_out_features):
    fs=SelectKBest(score_func=f_classif, k=k_out_features)
    df_sel=fs.fit_transform(X, y)
    if k_out_features=='all':
        for i in range(len(fs.scores_)):
            print('Feature of feat_sel_Num_to_Cat %s: %f' % (X.columns[i], fs.scores_[i]))
    #we have to create a dataframe
    cols=fs.get_support(indices=True)
    df_sel=X.iloc[:,cols]
    return df_sel

#Function to take a df with cat attributes and cat target and return df with k_out_features
def feat_sel_Cat_to_Cat_chi2(X, y, k_out_features):
    #chi-squared feature selection
    fs_chi2=SelectKBest(score_func=chi2, k=k_out_features)
    df_chi2=fs_chi2.fit_transform(X,y)
    if k_out_features=='all':
        for i in range(len(fs_chi2.scores_)):
            print('Feature of feat_sel_Cat_to_Cat chi2 %s: %f' % (X.columns[i], fs_chi2.scores_[i]))
    #we have to create a dataframe
    cols_chi2=fs_chi2.get_support(indices=True)
    df_chi2=X.iloc[:,cols_chi2]
        
    return df_chi2

def feat_sel_Cat_to_Cat_mutinf(X, y, k_out_features):
   
    #Mutual information feature selection
    fs_mutinf=SelectKBest(score_func=mutual_info_classif, k=k_out_features)
    df_mutinf=fs_mutinf.fit_transform(X,y)
    if k_out_features=='all':
        for i in range(len(fs_mutinf.scores_)):
            print('Feature of feat_sel_Cat_to_Cat mutual info %s: %f' % (X.columns[i], fs_mutinf.scores_[i]))
    cols_mutinf=fs_mutinf.get_support(indices=True)
    df_mutinf=X.iloc[:,cols_mutinf]
    return df_mutinf

#################
##Wrapper methods
#################

#RFE method with logistic regression or other specified estimator
def feat_sel_RFE(X,y,k_out_features=None, estimator='LogisticRegression'):
        
    #allows different kind of estimators
    if estimator=='LogisticRegression':
        model=LogisticRegression(solver='lbfgs', max_iter=2000)
    if estimator=='SVR':
        model=SVR(kernel='linear')
    
    #check the optimus number of output features for which the accuracy is highest
    #if k_out_features==None by default the number of output features is the half of total
    if k_out_features=='all':
        nof_list=np.arange(1,X.shape[1])            
        high_score=0
        #Variable to store the optimum features
        nof=0           
        score_list =[]
        from sklearn.model_selection import train_test_split
        for n in range(len(nof_list)):
            rfe = RFE(model,nof_list[n])
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
            X_train_rfe = rfe.fit_transform(X_train,y_train)
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe,y_train)
            score = model.score(X_test_rfe,y_test)
            score_list.append(score)
            if(score>high_score):
                high_score = score
                nof = nof_list[n]
        print("Optimum number of features: %d" %nof)
        print("Score with %d features: %f" % (nof, high_score))
        k_out_features=nof
                
    #obtain the pruned resultant df of features
    rfe=RFE(model,k_out_features)
    fit = rfe.fit(X, y)
    X_pruned=rfe.fit_transform(X,y)
    mask=fit.support_
    X_pruned=X.iloc[:,mask]
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    return X_pruned
        
#RFECV with the LogisticRegression estimator as default
def feat_sel_RFECV(X,y, estimator="LogisticRegression"):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    if estimator=='LogisticRegression':
        model=LogisticRegression(solver='lbfgs', max_iter=2000)
    if estimator=='SVR':
        model=SVR(kernel='linear')
    rfe=RFECV(model)    
    fit=rfe.fit(X, y)
    X_pruned=rfe.transform(X)
    mask=fit.support_
    X_pruned=X.iloc[:,mask]
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    return X_pruned

#Backward elimination
def feat_sel_backElimination(X,y):
#coded extracted from: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
    cols = list(X.columns)
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y,X_1).fit()
        p = pd.Series(model.pvalues.values[1:],index = cols)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    print(selected_features_BE)
    X_pruned=X[selected_features_BE]
    return X_pruned

#############
##Embedded methods
############
#Lasso linear model as regularizer adapted from https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
def feat_sel_Lasso(X,y):
    reg = Lasso()
    reg.fit(X, y)
    #print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    #print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)
    feat_sel=coef!=0
    X_pruned=X[feat_sel.index[feat_sel]]
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    return X_pruned

#Lasso linear model with CV as regularizer extracted from https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
def feat_sel_LassoCV(X,y):
    reg = LassoCV()
    reg.fit(X, y)
    #print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    #print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)
    feat_sel=coef!=0
    print('feat_sel in LassoCV: ', feat_sel)    
    X_pruned=X[feat_sel.index[feat_sel]]
    print("LassoCV picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    return X_pruned

#RidgeCV linear model as regularizer extracted and adapted from https://www.datacamp.com/community/tutorials/feature-selection-python
def feat_sel_RidgeCV(X,y):
    reg = RidgeCV()
    reg.fit(X, y)
    #print("Best alpha using built-in RidgeCV: %f" % reg.alpha_)
    #print("Best score using built-in RidgeCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)
    feat_sel=coef>=0
    X_pruned=X[feat_sel.index[feat_sel]]
    print("RidgeCV picked " + str(sum(coef > 0)) + " variables and eliminated the other " +  str(sum(coef <= 0)) + " variables")
    return X_pruned

#Ridge linear model as regularizer extracted and adapted from https://www.datacamp.com/community/tutorials/feature-selection-python
def feat_sel_Ridge(X,y):
    reg = Ridge()
    reg.fit(X, y)
    #print("Best alpha using built-in RidgeCV: %f" % reg.alpha_)
    #print("Best score using built-in RidgeCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)
    feat_sel=coef>=0
    X_pruned=X[feat_sel.index[feat_sel]]
    print("Ridge picked " + str(sum(coef > 0)) + " variables and eliminated the other " +  str(sum(coef <= 0)) + " variables")
    return X_pruned