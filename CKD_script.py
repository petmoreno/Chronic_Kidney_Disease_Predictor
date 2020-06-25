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


#import modules created 
import my_utils
import missing_val_imput



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

#Some fetures content seems to have the character \t.
#Let's remove such character for the sake of consistency
for i in range(0, len(df.columns)):           
    if df.dtypes[i]==np.object:        
        df.iloc[:,i] = df.iloc[:,i].str.replace(r'\t','')
        df.iloc[:,i] = df.iloc[:,i].str.replace(r' ','')        
#df_values(df)
        
#Lets convert pcv,wc and rc dtype to float64 dtype and if any strange character appears it turns to NAN
df['pcv']=pd.to_numeric(df['pcv'],errors='coerce')
df['wc']=pd.to_numeric(df['wc'],errors='coerce')   
df['rc']=pd.to_numeric(df['rc'],errors='coerce')

#Lets indicate numeric features
numerical_features=['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']

#Lets convert rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane to category
#also features sg, al, su will be set to category
features_to_category= ['sg','al','su','rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
for i in range(len(features_to_category)):
    df.loc[:,features_to_category[i]]=df.loc[:,features_to_category[i]].astype('category')
df.info()   

#lets convert as well the target feature to categorical value
df['classification']=df['classification'].astype('category')

#Understanding data phase is perform again
df.describe(include='all')

#Show the describe() of only numeric features
df.describe()

#Show the describe of category features
df.describe(include='category')

#Defining a strategy for handling the data missing. First we adopt
#complete case strategy where all NaN are cleaned by using drop_na
# Following onenote notebook of work plan for KCD 
########Step 1.a: Complete case
df_totalclean=df.dropna()
my_utils.info_adhoc(df_totalclean)

########Step 1.b: Imputation of missing values. 

missing_val_imput.print_df_threshold_shape(df)

#looking at the above result deciding on threshold=16 I would only lost 11 rows
#It seems reasonable a let the rest to impute.
df_totalclean_threshold= missing_val_imput.drop_threshold_info(df, 16)


########Step 1.b.i: Bayesian imputation for num attributes

#to decide an imputation estrategy lets look at hist of numerical attributes
pd.plotting.scatter_matrix(df_totalclean_threshold)
sn.pairplot(df_totalclean_threshold,hue='classification')

#numerical variables do not seem to have a normal distr then imputation strategy would be median


df_totalclean_threshold_imp_num=missing_val_imput.simpleImputeNum(df_totalclean_threshold,numerical_features,'median')

#########Step 1.b.ii: Bayesian imputation for category attributes

#first approach: assign unkown category to NaN values
df_totalclean_threshold_imp_cat_unk=missing_val_imput.simpleImputeCat(df_totalclean_threshold,features_to_category,'constant', 'unknown')

#second_approach:assign most_frequent strategy
df_totalclean_threshold_imp_cat_mostfq=missing_val_imput.simpleImputeCat(df_totalclean_threshold,features_to_category)


#creating the entire datasets witn simple imputation. 
#One for cat imputation of unknown
df_totalclean_threshold_imp_unk=pd.concat([df_totalclean_threshold_imp_num, df_totalclean_threshold_imp_cat_unk,df_totalclean_threshold['classification']], axis=1)
#other for cat imputation with mostfrequent
df_totalclean_threshold_imp_mostfq=pd.concat([df_totalclean_threshold_imp_num, df_totalclean_threshold_imp_cat_mostfq,df_totalclean_threshold['classification']], axis=1)



#########Step 1.b.iii.I: Multiple imputation with every features using IterativeImputer
## Numerical features 
df_totalclean_threshold_impMult_num=missing_val_imput.multipleImputNum(df_totalclean_threshold,numerical_features)

##Category features. PROBLEM:IterativeImputer does seem to work with category attributes
#df_totalclean_threshold_impMult_cat=missing_val_imput.multipleImputCat(df_totalclean_threshold,features_to_category)

#composing the entire data frame with IterativeImputer for num attr and SimpleImputer for cat attr.
df_totalclean_threshold_impMult=pd.concat([df_totalclean_threshold_impMult_num,df_totalclean_threshold_imp_cat_mostfq, df_totalclean_threshold['classification']], axis=1)

#########Step 1.b.iii.I: Multiple imputation with every features using KNNImputer
## Numerical features 
df_totalclean_threshold_impKNN_num=missing_val_imput.knnImputNum(df_totalclean_threshold,numerical_features)

# Category features. PROBLEM:KNNImputer does seem to work with category attributes
df_totalclean_threshold_impKNN_cat=missing_val_imput.knnImputCat(df_totalclean_threshold,features_to_category)

#composing the entire data frame with KNNImputer for num attr and SimpleImputer for cat attr.
df_totalclean_threshold_impKNN=pd.concat([df_totalclean_threshold_impKNN_num,df_totalclean_threshold_imp_cat_mostfq, df_totalclean_threshold['classification']], axis=1)

####Final of step1. The df created to feed feature selection are:
    # df_total_clean: complete case
    # df_totalclean_threshold_imp_unk: threshold set as 16 with unknown label as strategy imputation for cat attrib and median for num attr
    # df_totalclean_threshold_imp_mostfq: threshold set as 16 with most frequent as strategy imputation for cat attrib and median for num attr
    # df_totalclean_threshold_impMult:threshold set as 16 with most frequent as strategy imputation for cat attrib and multiple imputation for num attr
    # df_totalclean_threshold_impKNN:threshold set as 16 with most frequent as strategy imputation for cat attrib and KNNImputer for num attr
