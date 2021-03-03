# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:13:29 2021

@author: lizil
"""

import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve, classification_report
import h2o
from h2o.frame import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator
#Data Cleansing
##Merge two raw dataset
data = pd.read_csv('C:/Users/lizil/Dropbox/Job/Data/Fraud-Detection-master/Fraud_Data/Fraud_Data.csv',parse_dates = ['signup_time','purchase_time'])
data.head(10)
address_data = pd.read_csv('C:/Users/lizil/Dropbox/Job/Data/Fraud-Detection-master/Fraud_Data/IpAddress_to_Country.csv')
address_data.head()
countries = []
for i in range(len(data)):
    ip_address = data.loc[i,'ip_address']
    tmp = address_data[(address_data['lower_bound_ip_address']<=ip_address) & 
                      (address_data['upper_bound_ip_address']>=ip_address)]
    if len(tmp)==1:
        countries.append(tmp['country'].values[0])
    else:
        countries.append('NA')
data['countries'] = countries
data.head(10)
data.info()
## Outlier/Missing Data Detection
data.isnull().sum()
data.describe().transpose()
data.describe(include='object').transpose()
data['countries'].value_counts()
country_list = data['countries'].value_counts().to_frame().reset_index().rename(columns={'index':'country','countries':'count'})
main_country = list(country_list[country_list['count']>100]['country'])
main_country.remove('NA')
data['countries']=data['countries'].apply(lambda x: x if x in main_country else 'Other')
data.shape

#Feature Engineering
##Time Difference
### if time difference is too short, it is highly likely a fraud.
time_diff = data['purchase_time']-data['signup_time']
time_diff = time_diff.apply(lambda x: x.seconds)
data['time_diff']=time_diff

## #users per device
### if multiple users share same device, it is more likely a fraud activity
device_num = data[['user_id', 'device_id']].groupby('device_id').count().reset_index()
device_num = device_num.rename(columns={'user_id': 'device_num'})
device_num.head()
data = data.merge(device_num, how='left', on='device_id')

## #users per IP
### if multiple users share same IP, it is more likely a fraud activity
ip_num = data[['user_id', 'ip_address']].groupby('ip_address').count().reset_index()
ip_num = ip_num.rename(columns={'user_id': 'ip_num'})
data = data.merge(ip_num, how='left', on='ip_address')

## Day of week and week of year
data['signup_day'] = data['signup_time'].apply(lambda x: x.dayofweek)
data['signup_week'] = data['signup_time'].apply(lambda x: x.week)
data['purchase_day'] = data['purchase_time'].apply(lambda x: x.dayofweek)
data['purchase_week'] = data['purchase_time'].apply(lambda x: x.week)
data.head()

#Exploratory Data Analysis

##Correlation Analysis
corr = data[data.columns].corr()
plt.figure(figsize = (30,20))
sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f")

## Target VS Categorical Data
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
sns.barplot(x='source', y='class', data=data, ax=ax[0,0])
ax[0,0].set_title('Fraud Rate by Source', fontsize=16)
sns.barplot(x='browser', y='class', data=data, ax=ax[0,1])
ax[0,1].set_title('Fraud Rate by Browser', fontsize=16)
sns.barplot(x='sex', y='class', data=data, ax=ax[1,0])
ax[1,0].set_title('Fraud Rate by Gender', fontsize=16)
sns.barplot(x='purchase_day', y='class', data=data, ax=ax[1,1])
ax[1,1].set_title('Fraud Rate by Purchase Day of the Week', fontsize=16)
ax[1,1].set_xticks(np.arange(7))
ax[1,1].set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])

plt.tight_layout()
plt.show()

## Target VS Continuous Data
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(18,12))
sns.lineplot(x='age',y='class',data=data,ax=ax[0,0])
ax[0,0].set_title('Mean Fraud Rate by Age', fontsize = 16)
ax[0,0].set_ylabel('Mean Fraud Rate')
ax[0,0].grid()
sns.lineplot(x='purchase_value',y='class',data=data,ax=ax[0,1])
ax[0,1].set_title('Mean Fraud Rate by Purchase Value', fontsize = 16)
ax[0,1].set_ylabel('Mean Fraud Rate')
ax[0,1].grid()
sns.lineplot(x='device_num',y='class',data=data,ax=ax[1,0])
ax[1,0].set_title('Mean Fraud Rate by Device Number', fontsize = 16)
ax[1,0].set_ylabel('Mean Fraud Rate')
ax[1,0].grid()
sns.lineplot(x='ip_num',y='class',data=data,ax=ax[1,1])
ax[1,1].set_title('Mean Fraud Rate by IP Number', fontsize = 16)
ax[1,1].set_ylabel('Mean Fraud Rate')
ax[1,1].grid()
plt.show()

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(20,8))
sns.lineplot(x='signup_week',y='class',data=data,ax=ax[0])
ax[0].set_title('Mean Fraud Rate by SignUp Week', fontsize = 16)
ax[0].set_ylabel('Mean Fraud Rate')
ax[0].grid()
sns.lineplot(x='purchase_week',y='class',data=data,ax=ax[1])
ax[1].set_title('Mean Fraud Rate by Purchase Week', fontsize = 16)
ax[1].set_ylabel('Mean Fraud Rate')
ax[1].grid()

plt.show()

# Random Forest Model Fitting
##Select model features and targets
columns = ['signup_day', 'signup_week', 'purchase_day', 'purchase_week', 'purchase_value', 'source', 
           'browser', 'sex', 'age', 'countries', 'time_diff', 'device_num', 'ip_num', 'class']
data = data[columns]
data.head()

## Initiate the model
h2o.init()
h2o.remove_all()

##Categorize all factor columns
h2o_df = H2OFrame(data)

for name in ['signup_day', 'purchase_day', 'source', 'browser', 'sex', 'countries', 'class']:
    h2o_df[name] = h2o_df[name].asfactor()
h2o_df.summary()

##Split the train/test data
strat_split = h2o_df['class'].stratified_split(test_frac=0.3, seed=42)
train = h2o_df[strat_split == 'train']
test = h2o_df[strat_split == 'test']

##Split the feature/target
feature = ['signup_day', 'signup_week', 'purchase_day', 'purchase_week', 'purchase_value', 
           'source', 'browser', 'sex', 'age', 'countries', 'time_diff', 'device_num', 'ip_num']
target = 'class'

##Build model
model = H2ORandomForestEstimator(balance_classes=True, ntrees=100, mtries=-1, stopping_rounds=5, 
                                 stopping_metric='auc', score_each_iteration=True, seed=42)
model.train(x=feature, y=target, training_frame=train, validation_frame=test)
model.score_history()

##Show feature importance
importance = model.varimp(use_pandas=True)
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='scaled_importance', y='variable', data=importance)
plt.show()

##AUC-ROC Curve
train_true = train.as_data_frame()['class'].values
test_true = test.as_data_frame()['class'].values
## to predict the target on certain dataset in H2O, just use model.predict(dataset), 'p0' is the probability of class 0, and 
## "p1" is the probability of class 1
train_pred = model.predict(train).as_data_frame()['p1'].values
test_pred = model.predict(test).as_data_frame()['p1'].values
train_fpr, train_tpr, _ = roc_curve(train_true, train_pred)
test_fpr, test_tpr, _ = roc_curve(test_true, test_pred)
train_auc = np.round(auc(train_fpr, train_tpr), 3)
test_auc = np.round(auc(test_fpr, test_tpr), 3)
print(classification_report(y_true=test_true, y_pred=(test_pred > 0.5).astype(int)))

train_fpr = np.insert(train_fpr, 0, 0)
train_tpr = np.insert(train_tpr, 0, 0)
test_fpr = np.insert(test_fpr, 0, 0)
test_tpr = np.insert(test_tpr, 0, 0)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_fpr, train_tpr, label='Train AUC: ' + str(train_auc))
ax.plot(test_fpr, test_tpr, label='Test AUC: ' + str(test_auc))
ax.plot(train_fpr, train_fpr, 'k--', label='Chance Curve')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.grid(True)
ax.legend(fontsize=12)
plt.show()

cols = ['device_num', 'time_diff', 'ip_num','countries']
_ = model.partial_plot(data=train, cols=cols, nbins=200, figsize=(18, 20))
