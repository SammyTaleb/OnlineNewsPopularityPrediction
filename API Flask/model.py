import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.metrics import *
from sklearn.ensemble import *
from numpy import array

#Data Importing and Pre-Processing

dataset=pd.read_csv('OnlineNewsPopularity.csv')
dataset=dataset.drop_duplicates()
dataset.columns = dataset.columns.str.lstrip()
dataset=dataset.drop(['url','timedelta'],axis=1)
dataset=dataset.dropna()
dataset=dataset.drop(['kw_avg_min','n_unique_tokens','n_non_stop_words'],axis=1)
dataset['sharesBool']=dataset['shares']>1400

#Data Splitting
train,test=train_test_split(dataset,test_size=0.2,random_state=0)
train_Y=train['shares']
train_Y_bool=train['sharesBool']
train_X=train.drop(['shares','sharesBool'],axis=1)
test_Y=test['shares']
test_Y_bool=test['sharesBool']
test_X=test.drop(['shares','sharesBool'],axis=1)


#Building Models
best_params={'bootstrap': True, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 500}
clf_rf=RandomForestClassifier(**best_params)
clf_rf.fit(train_X,train_Y_bool)



#pickle.dump(clf_rf, open('clf_rf.pkl','wb'))
