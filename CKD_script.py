# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## All necessary modules as well as different functions that will be used in this work are explicit here.
#import all neccesary modules
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

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


#############################
##Step 1: Mispelling correction
#############################
#Some mispelling or wrong character should be removed
# for i in range(0, len(df.columns)):           
#     if df.dtypes[i]==np.object:        
#         df.iloc[:,i] = df.iloc[:,i].str.replace(r'\t','')
#         df.iloc[:,i] = df.iloc[:,i].str.replace(r' ','')

train_set_copy=train_set.copy()
train_set_corrected=adhoc_transf.mispelling(train_set)
        

#############################
##Step 2: Transform num and cat features
#############################
        
#Lets convert pcv,wc and rc dtype to float64 dtype and if any strange character appears it turns to NAN
# df['pcv']=pd.to_numeric(df['pcv'],errors='coerce')
# df['wc']=pd.to_numeric(df['wc'],errors='coerce')   
# df['rc']=pd.to_numeric(df['rc'],errors='coerce')

#Lets indicate numeric features
numerical_features=['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
train_set_corrected=adhoc_transf.num_feat_cast(train_set_corrected, numerical_features)


#Lets convert rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane to category
#also features sg, al, su will be set to category
features_to_category= ['sg','al','su','rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
train_set_corrected=adhoc_transf.cat_feat_cast(train_set_corrected, features_to_category)


#lets convert as well the target feature to categorical value
train_set_corrected= adhoc_transf.target_to_cat(train_set_corrected,'classification')
train_set_corrected.info()   


# #Understanding data phase is perform again
# df.describe(include='all')

# #Show the describe() of only numeric features
# df.describe()

# #Show the describe of category features
# df.describe(include='category')


    

###########################
#Step 3 Data Missing
###########################

#Defining a strategy for handling the data missing. First we adopt
#complete case strategy where all NaN are cleaned by using drop_na
# Following onenote notebook of pipeline for CKD 
########Step 3.a: Complete case
df_totalclean=train_set.dropna()
my_utils.info_adhoc(df_totalclean)

########Step 3.b: Imputation of missing values. 

missing_val_imput.print_df_threshold_shape(train_set)

#looking at the above result deciding on threshold=16 I would only lost 11 rows
#It seems reasonable a let the rest to impute.
df_totalclean_threshold= missing_val_imput.drop_threshold_info(train_set, 16)

########Step 3.b.i: Bayesian imputation for num attributes

#to decide an imputation estrategy lets look at hist of numerical attributes
#pd.plotting.scatter_matrix(df_totalclean_threshold)
#sn.pairplot(df_totalclean_threshold,hue='classification')

#numerical variables do not seem to have a normal distr then imputation strategy would be median


df_totalclean_threshold_imp_num=missing_val_imput.simpleImputeNum(df_totalclean_threshold,numerical_features,'median')

#########Step 3.b.ii: Bayesian imputation for category attributes

#first approach: assign unkown category to NaN values
df_totalclean_threshold_imp_cat_unk=missing_val_imput.simpleImputeCat(df_totalclean_threshold,features_to_category+['classification'],'constant', 'unknown')

#second_approach:assign most_frequent strategy
df_totalclean_threshold_imp_cat_mostfq=missing_val_imput.simpleImputeCat(df_totalclean_threshold,features_to_category+['classification'])

#creating the entire datasets witn simple imputation. 
#One for cat imputation of unknown
df_totalclean_threshold_imp_unk=pd.concat([df_totalclean_threshold_imp_num, df_totalclean_threshold_imp_cat_unk], axis=1)
#other for cat imputation with mostfrequent
df_totalclean_threshold_imp_mostfq=pd.concat([df_totalclean_threshold_imp_num, df_totalclean_threshold_imp_cat_mostfq], axis=1)



#########Step 3.b.iii.I: Multiple imputation with every features using IterativeImputer
## Numerical features 
df_totalclean_threshold_impMult_num=missing_val_imput.multipleImputNum(df_totalclean_threshold,numerical_features)

##Category features. PROBLEM:IterativeImputer does seem to work with category attributes
#df_totalclean_threshold_impMult_cat=missing_val_imput.multipleImputCat(df_totalclean_threshold,features_to_category)

#composing the entire data frame with IterativeImputer for num attr and SimpleImputer for cat attr.
df_totalclean_threshold_impMult=pd.concat([df_totalclean_threshold_impMult_num,df_totalclean_threshold_imp_cat_mostfq], axis=1)

#########Step 3.b.iii.I: Multiple imputation with every features using KNNImputer
## Numerical features 
df_totalclean_threshold_impKNN_num=missing_val_imput.knnImputNum(df_totalclean_threshold,numerical_features)

# Category features. PROBLEM:KNNImputer does seem to work with category attributes
df_totalclean_threshold_impKNN_cat=missing_val_imput.knnImputCat(df_totalclean_threshold,features_to_category)

#composing the entire data frame with KNNImputer for num attr and SimpleImputer for cat attr.
df_totalclean_threshold_impKNN=pd.concat([df_totalclean_threshold_impKNN_num,df_totalclean_threshold_imp_cat_mostfq], axis=1)

####Final of step3. The df created to feed feature selection are:
    # df_total_clean: complete case
    # df_totalclean_threshold_imp_unk: threshold set as 16 with unknown label as strategy imputation for cat attrib and median for num attr
    # df_totalclean_threshold_imp_mostfq: threshold set as 16 with most frequent as strategy imputation for cat attrib and median for num attr
    # df_totalclean_threshold_impMult:threshold set as 16 with most frequent as strategy imputation for cat attrib and multiple imputation for num attr
    # df_totalclean_threshold_impKNN:threshold set as 16 with most frequent as strategy imputation for cat attrib and KNNImputer for num attr


#############################
####Step 4: Feature Encoding
#############################
#For a first try we will use df_totalclean_threshold_imp_mostfq
#We have to map categorical attributes to num codes. Using ordinal encoder and label encoder

df_cat_coded= df_totalclean_threshold_imp_mostfq.copy()
oe=OrdinalEncoder()
df_cat_coded[features_to_category]=oe.fit_transform(df_cat_coded[features_to_category])
df_cat_coded[features_to_category]
le=LabelEncoder()
df_cat_coded['classification']=le.fit_transform(df_cat_coded['classification'])
my_utils.info_adhoc(df_cat_coded)

#############################
####Step 5: Feature Selection
#############################
X=df_cat_coded.drop('classification',axis=1)
X_num=df_cat_coded[numerical_features]
X_cat=df_cat_coded[features_to_category]
y=df_cat_coded['classification']

#Step 5.1: Applying filtering methods of feature selection
df_featSel_num=feature_select.feat_sel_Num_to_Cat(X_num,y,'all')

#Applying ANOVA for numeric attributes
df_featSel_num=feature_select.feat_sel_Num_to_Cat(X_num,y,5)
df_featSel_num.corr()

#Applying chi2 for category attributes
df_featSel_cat_chi2=feature_select.feat_sel_Cat_to_Cat_chi2(X_cat,y,'all')
df_featSel_cat_chi2=feature_select.feat_sel_Cat_to_Cat_chi2(X_cat,y,5)

#Applying mutual information for all attributes
df_featSel_cat_mutinf=feature_select.feat_sel_Cat_to_Cat_mutinf(X,y,'all')
df_featSel_cat_mutinf=feature_select.feat_sel_Cat_to_Cat_mutinf(X,y,10)

df_cat_coded_featSel_chi2=pd.concat([df_featSel_num,df_featSel_cat_chi2,y], axis=1)
df_cat_coded_featSel_mutinf=pd.concat([df_featSel_cat_mutinf,y], axis=1)

#Step 5.2: Applying wrapper methods for feature selection
#Applying RFE with an automatic detector of output features number.
df_cat_coded_featSel_RFE=feature_select.feat_sel_RFE(X,y,k_out_features='all')
df_cat_coded_featSel_RFE=pd.concat([df_cat_coded_featSel_RFE,y], axis=1)

#Applying RFECV to check the optimal number of features
df_cat_coded_featSel_RFECV=feature_select.feat_sel_RFECV(X,y)
df_cat_coded_featSel_RFECV=pd.concat([df_cat_coded_featSel_RFECV,y], axis=1)

#Applying Backward Elimination 
df_cat_coded_featSel_BackElim=feature_select.feat_sel_backElimination(X,y)
df_cat_coded_featSel_BackElim=pd.concat([df_cat_coded_featSel_BackElim,y], axis=1)

#Step 5.3: Applying embedded methods for feature selection
#Applying LassoCV regularization
df_cat_coded_featSel_LassoCV=feature_select.feat_sel_LassoCV(X,y)
df_cat_coded_featSel_LassoCV=pd.concat([df_cat_coded_featSel_LassoCV,y], axis=1)

#Applying RidgeCV regularization
df_cat_coded_featSel_RidgeCV=feature_select.feat_sel_RidgeCV(X,y)
df_cat_coded_featSel_RidgeCV=pd.concat([df_cat_coded_featSel_RidgeCV,y], axis=1)

####Final of step5. The df created to feed normalization and standarization are:
# df_cat_coded_featSel_chi2   
# df_cat_coded_featSel_mutinf
# df_cat_coded_featSel_RFE
# df_cat_coded_featSel_RFECV
# df_cat_coded_featSel_BackElim
# df_cat_coded_featSel_LassoCV
# df_cat_coded_featSel_RidgeCV

#############################
####Step 6: Pre-processing to normalize/standardize features
#############################


#Category attributes are ordinal or binomial.
#Ordinal category attributes have been coded already before feature selection.
#There is no need to one-hot-encoding
num_feat_remain=preprocessing.get_numerical_columns(df_cat_coded_featSel_RFECV,numerical_features)
df_cat_coded_featSel_RFECV[num_feat_remain]=preprocessing.minmaxscaler(df_cat_coded_featSel_RFECV[num_feat_remain])


#Separation to X_train and y_train
X_train=df_cat_coded_featSel_RFECV.drop('classification', axis=1)
y_train=df_cat_coded_featSel_RFECV['classification'].copy()

# df_cat_coded= df_totalclean.copy()
# oe=OrdinalEncoder()
# df_cat_coded[features_to_category]=oe.fit_transform(df_cat_coded[features_to_category])
# df_cat_coded[features_to_category]
# le=LabelEncoder()
# df_cat_coded['classification']=le.fit_transform(df_cat_coded['classification'])
# my_utils.info_adhoc(df_cat_coded)
# pd.plotting.scatter_matrix(df_cat_coded)
# sn.pairplot(df_cat_coded,hue='classification')

# X_train, y_train=preprocessing.preprocessing(df_cat_coded,'classification',num_feat=numerical_features)

#############################
####Step 7: Apply ML models
#############################
#Several classifier with Cross validation will be applied
from sklearn.model_selection import cross_val_score
#Classifier models to use
from sklearn.linear_model import SGDClassifier
sgd_clf=SGDClassifier()
cross_val_score(sgd_clf,X_train,y_train, cv=5, scoring='accuracy')


from sklearn.linear_model import LogisticRegression
logreg_clf=LogisticRegression()
cross_val_score(logreg_clf,X_train,y_train, cv=5, scoring='accuracy')

from sklearn.svm import LinearSVC
linsvc_clf=LinearSVC()
cross_val_score(linsvc_clf,X_train,y_train, cv=5, scoring='accuracy')

from sklearn.svm import SVC
svc_clf=SVC()
cross_val_score(svc_clf,X_train,y_train, cv=5, scoring='accuracy')

from sklearn.tree import DecisionTreeClassifier
dectree_clf=DecisionTreeClassifier()
cross_val_score(dectree_clf,X_train,y_train, cv=5, scoring='accuracy')

from sklearn.ensemble import RandomForestClassifier
rndforest_clf=RandomForestClassifier()
cross_val_score(rndforest_clf,X_train,y_train, cv=5, scoring='accuracy')

#




