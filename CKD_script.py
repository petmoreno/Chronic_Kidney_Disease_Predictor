# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## All necessary modules as well as different functions that will be used in this work are explicit here.
#import all neccesary modules
import pandas as pd
import numpy as np
import arff
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer



#%matplotlib inline 
def df_values(df):
    for i in range(0, len(df.columns)):
        print("*****start of feature ", df.columns[i], "*************************")
        print (df.iloc[:,i].value_counts())       
        print ("*****end of feature ", df.columns[i], "************************** \n")

def info_adhoc(df):
    d=(df.count()/len(df))*100
    df_info=pd.DataFrame(data=d, columns=['% non-null values'])
    df_info['non-null values']=df.count()
    df_info['dtype']=df.dtypes
    return df_info

#importing file into a pandas dataframe# As being unable to extract data from it original source, the csv file is downloaded from
#https://www.kaggle.com/mansoordaku/ckdisease
path_data=r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\Chronic_Kidney_Disease\kidney_disease.csv'
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
info_adhoc(df)

#As seen above, there are some strange caracters in pcv feature, therefore we will explore every features' value to homogeneize it.
df_values(df)

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
info_adhoc(df_totalclean)

########Step 1.b: Imputation of missing values. 
for i in range (len(df.columns)+1):
    #df_partialclean=df.dropna(thresh=i)
    print('Shape of df_partialclean with threshold {} is: {}'.format(i,str(df.dropna(thresh=i).shape[0])))

#looking at the above result deciding on threshold=16 I would only lost 11 rows
#It seems reasonable a let the rest to impute.
df_totalclean_threshold=df.dropna(thresh=16)
info_adhoc(df_totalclean_threshold)

########Step 1.b.i: Bayesian imputation for num attributes

#to decide an imputation estrategy lets look at hist of numerical attributes
pd.plotting.scatter_matrix(df_totalclean_threshold)
sn.pairplot(df_totalclean_threshold,hue='classification')

#numerical variables do not seem to have a normal distr then imputation strategy would be median
numerical_features=['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
df_totalclean_threshold_imp_num=df_totalclean_threshold[numerical_features]
num_imputer=SimpleImputer(strategy='median')
df_imp_num=num_imputer.fit_transform(df_totalclean_threshold_imp_num)
df_totalclean_threshold_imp_num=pd.DataFrame(df_imp_num, columns=df_totalclean_threshold_imp_num.columns)

#########Step 1.b.ii: Bayesian imputation for category attributes
df_totalclean_threshold_imp_cat=df_totalclean_threshold[features_to_category]
#first approach: assign unkown category to NaN values
cat_unk_imputer=SimpleImputer(strategy='constant', fill_value='unknown')
imp_cat_unk=cat_unk_imputer.fit_transform(df_totalclean_threshold_imp_cat)
df_totalclean_threshold_imp_cat_unk=pd.DataFrame(imp_cat_unk, columns=df_totalclean_threshold_imp_cat.columns)


#second_approach:assign most_frequent strategy

cat_mostfq_imputer=SimpleImputer(strategy='most_frequent')
imp_cat_mostfq=cat_mostfq_imputer.fit_transform(df_totalclean_threshold_imp_cat)
df_totalclean_threshold_imp_cat_mostfq=pd.DataFrame(imp_cat_mostfq, columns=df_totalclean_threshold_imp_cat.columns)


#composing the entire dataframe by concating num and cat attributes
df_totalclean_threshold_imp_unk=pd.concat([df_totalclean_threshold_imp_num,df_totalclean_threshold_imp_cat_unk,df_totalclean_threshold['classification']], axis=1)
df_totalclean_threshold_imp_mostfq=pd.concat([df_totalclean_threshold_imp_num,df_totalclean_threshold_imp_cat_mostfq, df_totalclean_threshold['classification']], axis=1)


#########Step 1.b.iii.I: Multiple imputation with every features using IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
## try only with numerical features 
df_totalclean_threshold_impMult_num=df_totalclean_threshold[numerical_features]
imp_mult_num=IterativeImputer(random_state=0, sample_posterior='True')#set sample_posterior='True' for using to multiple imputation
impMult_num=imp_mult_num.fit_transform(df_totalclean_threshold_imp_num)
df_totalclean_threshold_impMult_num=pd.DataFrame(impMult_num, columns=df_totalclean_threshold_impMult_num.columns)

##and then only with category. PROBLEM:IterativeImputer does seem to work with category attributes
#Convert categories to numbers
# df_totalclean_threshold_impMult_cat=df_totalclean_threshold[features_to_category]
# for i in range(len(features_to_category)):
#     df_totalclean_threshold_impMult_cat.loc[:,features_to_category[i]]=df_totalclean_threshold_impMult_cat.loc[:,features_to_category[i]].cat.codes

# #The NaN has been coded to -1. So we have to revert it for the imputer
# df_totalclean_threshold_impMult_cat.replace(to_replace=-1,value=np.NAN,inplace=True)
# df_totalclean_threshold_impMult_cat.head()
# #Apply iterative imputer
# #****Review
# imp_mult_cat=IterativeImputer(initial_strategy='most_frequent',random_state=0, sample_posterior='True')#set sample_posterior='True' for using to multiple imputation)
# impMult_cat=imp_mult_cat.fit_transform(df_totalclean_threshold_impMult_cat)
# df_totalclean_threshold_impMult_cat=pd.DataFrame(impMult_cat, columns=df_totalclean_threshold_impMult_cat.columns)


#composing the entire data frame with IterativeImputer for num attr and SimpleImputer for cat attr.
df_totalclean_threshold_impMult=pd.concat([df_totalclean_threshold_impMult_num,df_totalclean_threshold_imp_cat_mostfq, df_totalclean_threshold['classification']], axis=1)

#########Step 1.b.iii.I: Multiple imputation with every features using KNNImputer
from sklearn.impute import KNNImputer


## try only with numerical features 
df_totalclean_threshold_impKNN_num=df_totalclean_threshold[numerical_features]
knn_imp=KNNImputer()
impKNN_num=knn_imp.fit_transform(df_totalclean_threshold_imp_num)
df_totalclean_threshold_impKNN_num=pd.DataFrame(impMult_num, columns=df_totalclean_threshold_impKNN_num.columns)

# ##and then only with category. PROBLEM:KNNImputer does seem to work with category attributes
# #Convert categories to numbers
# df_totalclean_threshold_impKNN_cat=df_totalclean_threshold[features_to_category]
# for i in range(len(features_to_category)):
#     df_totalclean_threshold_impKNN_cat.loc[:,features_to_category[i]]=df_totalclean_threshold_impKNN_cat.loc[:,features_to_category[i]].cat.codes

# #The NaN has been coded to -1. So we have to revert it for the imputer
# df_totalclean_threshold_impKNN_cat.replace(to_replace=-1,value=np.NAN,inplace=True)

# #****Review
# #Apply KNN imputer
# impKNN_cat=KNNImputer()
# impKNN_cat_arr=impKNN_cat.fit_transform(df_totalclean_threshold_impKNN_cat)
# df_totalclean_threshold_impKNN_cat=pd.DataFrame(impKNN_cat_arr, columns=df_totalclean_threshold_impKNN_cat.columns)

#composing the entire data frame with KNNImputer for num attr and SimpleImputer for cat attr.
df_totalclean_threshold_impKNN=pd.concat([df_totalclean_threshold_impKNN_num,df_totalclean_threshold_imp_cat_mostfq, df_totalclean_threshold['classification']], axis=1)

####Final of step1. The df created to feed feature selection are:
    # df_total_clean: complete case
    # df_totalclean_threshold_imp_unk: threshold set as 16 with unknown label as strategy imputation for cat attrib and median for num attr
    # df_totalclean_threshold_imp_mostfq: threshold set as 16 with most frequent as strategy imputation for cat attrib and median for num attr
    # df_totalclean_threshold_impMult:threshold set as 16 with most frequent as strategy imputation for cat attrib and multiple imputation for num attr
    # df_totalclean_threshold_impKNN:threshold set as 16 with most frequent as strategy imputation for cat attrib and KNNImputer for num attr
