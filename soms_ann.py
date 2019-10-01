# -*- coding: utf-8 -*-
"""
Created on Sun Oct 07 09:20:40 2018
"""


import os
import sys
os.path.abspath('C:\Users\\meryl\\Desktop\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\meryl\\Documents\updews-pycodes-master\Analysis')) #GroundDataAlertLib - line 21
import querySenslopeDb as q
import matplotlib.pyplot as plt
import pandas as pd
import mysql.connector as sql
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve, auc
import cPickle

db_connection = sql.connect(host='192.168.150.129', database='senslopedb', user='dyna_staff', password='accelerometer')


def rainfall (start_time, end_time, gauge):
    df = pd.read_sql("SELECT * FROM senslopedb.{}".format(gauge), con=db_connection)
    df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    df.sort_values(['timestamp'],inplace = True)
    df['timestamp']=df['timestamp'].dt.round('30min')
    col = ['name', 'batv1', 'batv2', 'cur', 'boostv1', 'boostv2', 'charge','csq', 'temp', 'hum', 'flashp']
    df.drop(col,axis=1, inplace =True)
    df ['roll_sum_1d'] = df['r15m'].rolling(48).sum()
    df['norm'] = (df['roll_sum_1d'] - df['roll_sum_1d'].min()) / (df['roll_sum_1d'].max() - df['roll_sum_1d'].min())
    rainfall = df.reset_index(drop=True)
    
    return rainfall

def soms (site, start, end):
    df = q.GetSomsData(site, start, end)
    df['ts']=df['ts'].dt.round('30min')
    df = df[df.id < 7]
    df = df[df.msgid == 26]
    df = df.groupby('ts')['mval1'].mean()
    df = pd.DataFrame({'timestamp':df.index, 'mval1':df.values})
    
    return df

def train_test(soms, rainfall, ave_window = '1H'):
    final = pd.merge(soms, rainfall, on = 'timestamp')
    final['dif_s'] = final.mval1 - final.mval1.shift(1)
    final['dif_r'] = final.r15m - final.r15m.shift(1)
    
    rain = final[['timestamp','r15m']].resample(ave_window, label = 'right',closed = 'right',on = 'timestamp').sum().r15m.dropna().values
    soms = final[['timestamp','mval1']].resample(ave_window, label = 'right',closed = 'right',on = 'timestamp').sum().mval1.dropna().values
    soms2 = (soms - min(soms))/(max(soms) - min(soms))
    rain2 = (rain - min(rain))/(max(rain) - min(rain))
   
    m = 3
    r = []
    s= []
        
    #### Iterate through the data frame based on m
    for i in range(len(rain)-m):
        #### Add to list
        r.append(rain2[i:i+m])
        s.append(soms2[i+m])
    
    #### Convert results to array
    r = np.array(r)
    s = np.array(s)
    
    
    X_train, X_test, y_train, y_test = train_test_split(r, s, train_size=0.90)
    scaler = StandardScaler()

    # Fit only to the training data
    scaler.fit(X_train)


    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


#    last_index = int(len(r)*0.75)
#    X_train = r[0:last_index]
#    X_test = r[last_index:]
#    y_train = s[0:last_index]
#    y_test = s[last_index:]
#    
    y_train = np.around(y_train, decimals = 1)
    y_train = y_train.astype(str)
    y_test = np.around(y_test, decimals = 1)
    y_test = y_test.astype(str)
    
#    X_train = X_train.reshape(len(X_train),1)
#    X_test = X_test.reshape(len(X_test),1)


    return X_train, X_test, y_train, y_test

def GetOptimalCLF2(train_x,train_y,rand_starts):    

    min_loss = 1e10
    
    #### Perform number of trainings according to random start set
    for i in range(rand_starts):
        
        #### Print current status
        print "Iteration number {}".format(i+1)
        
        #### Initialize ANN network
        clf = MLPClassifier(hidden_layer_sizes=(35,35,35), solver = 'lbfgs', early_stopping = True, 
                            max_iter = 1000000000000,tol = 1e-10, validation_fraction = 0.30, learning_rate_init=0.00000001)
    
        #### Fit data
        clf.fit(train_x,train_y)
        
        #### Get current loss
        cur_loss = clf.loss_
        print 'min_loss = ', min_loss
        print 'cur_loss = ', cur_loss
        
        #### Save current clf if loss is minimum
        if cur_loss < min_loss:
            
            #### Set min_loss to a new value
            min_loss = cur_loss
            
            #### Set max_clf to new value
            max_clf = clf
    
    return max_clf

def denormalize(y_train, y_test, pred_test, pred_train, soms):
    n = []
    for i in y_train:
        n.append(float(i))
    n = np.array(n)
    train = n
    
    m = []
    
    for i in y_test:
        m.append(float(i))
    m = np.array(m)
    test = m
    
    ptrain = []
    for i in pred_train:
        ptrain.append(float(i))
    p_train = np.array(ptrain)
    
    ptest = []
    for i in pred_test:
        ptest.append(float(i))
    p_test = np.array(ptest)
    
    df = soms[['timestamp','mval1']].resample('4H', label = 'right',closed = 'right',on = 'timestamp').mean().mval1.dropna().values
    actual = np.concatenate((train,test))*(max(df) - min(df)) + min(df)
    pred = np.concatenate((p_train,p_test))*(max(df) - min(df)) + min(df)

    return actual, pred 

start = '2018-04-24'
end = '2018-06-15'
site = 'nagsam'
gauge = 'nagtbw'

soms = soms(site, start, end)
rainfall = rainfall(start, end, gauge)

X_train, X_test, y_train, y_test = train_test(soms, rainfall)

clf = GetOptimalCLF2(X_train,y_train,rand_starts = 10)
#
clf.fit(X_train,y_train)

pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)

#
#actual, pred = denormalize(y_train, y_test, pred_test, pred_train, soms)
#
fig, (ax) = plt.subplots(nrows = 2, sharex = True)

ax[0].plot(pred_train, color = 'red', label = 'predicted')
ax[0].legend()
ax[1].plot(y_train, color = 'blue', label = 'actual')
ax[1].legend()

###RMSE
pred_test = pred_test.astype(float)
y_test = y_test.astype(float)

pred_train = pred_train.astype(float)
y_train = y_train.astype(float)


#print(confusion_matrix(y_test,pred_train))
#print(classification_report(y_test,pred_train))




#############ROC
tot =[]
pos = []
neg = []
for i in pred_train:
    for j in y_train:
        if i == j: ################True positive
            val = 1
            tot.append(val)
            pos.append(val)
        else: ####################True negative
            val = 0
            tot.append(val)
            neg.append(val)
            
recall = float(len(pos)) / (float(len(pos)) + float(len(neg)))
print 'recall', recall * 100, '%'


# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)    

# load it again
#with open('my_dumped_classifier.pkl', 'rb') as fid:
#    clf = cPickle.load(fid)
#    