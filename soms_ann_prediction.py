# -*- coding: utf-8 -*-
"""
Created on Tue Oct 09 13:26:28 2018
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
import cPickle
import matplotlib.dates as md

db_connection = sql.connect(host='192.168.150.129', database='senslopedb', user='dyna_staff', password='accelerometer')




def rainfall (start, end, gauge):
    df = pd.read_sql("SELECT * FROM senslopedb.{}".format(gauge), con=db_connection)
    df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    df.sort_values(['timestamp'],inplace = True)
    df['timestamp']=df['timestamp'].dt.round('30min')
    df.drop(['name', 'batv1', 'batv2', 'cur', 'boostv1', 'boostv2', 'charge','csq', 'temp', 'hum', 'flashp'],axis=1, inplace =True)
#    df['norm'] = (df['r15m'] - df['r15m'].min()) / (df['r15m'].max() - df['r15m'].min())
    df ['roll_sum_1d'] = df['r15m'].rolling(48).sum()
    df['norm'] = (df['roll_sum_1d'] - df['roll_sum_1d'].min()) / (df['roll_sum_1d'].max() - df['roll_sum_1d'].min())
#    df ['roll_sum_3d'] = df['r15m'].rolling(144).sum()
    rainfall = df.reset_index(drop=True)
    
    return rainfall

def soms (site, start, end):
    df = q.GetSomsData(site, start, end)
    df['ts']=df['ts'].dt.round('30min')
    df = df[df.id < 7]
    df = df[df.msgid == 21]
    df = df.groupby('ts')['mval1'].mean()
    df = pd.DataFrame({'timestamp':df.index, 'mval1':df.values})
    
    return df

start = '2018-02-13'
end = '2018-08-18'
site = 'nagsam'
gauge = 'nagtbw'

rainfall = rainfall(start, end, gauge)
soms = soms(site, start, end)

rain = rainfall[['timestamp','r15m']].resample('1H', label = 'right',closed = 'right',on = 'timestamp').sum().r15m.dropna().values
time = rainfall[['timestamp','r15m']].resample('1H', label = 'right',closed = 'right',on = 'timestamp').sum().r15m.dropna()
time = time.index

               
rain2 = (rain - min(rain))/(max(rain) - min(rain))
m = 3
r = []
t = []

for i in range(len(rain2)-m):
    r.append(rain[i:i+m])
    t.append(time[i+m])
    
r = np.array(r)
t = np.array(t)

t = t.reshape(len(t),1)
rain2 = rain2.reshape(len(rain2),1)

#load clf
with open('my_dumped_classifier.pkl', 'rb') as fid:
    clf = cPickle.load(fid)
predicted = clf.predict(r)


fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle('Predicted SOMS from {} to {}'.format(start,end), fontsize = 20)
ax.plot(t, predicted, color = 'blue', label = 'predicted')
ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
ax.set_ylabel('SOMS (normalized)', color='b', fontsize = 20)
ax.set_xlabel('timestamp', fontsize = 20)
ax2 = ax.twinx()
ax2.plot(rainfall.timestamp, rainfall.r15m, color = 'red', label = 'rainfall')
ax2.set_ylabel('rainfall (mm)', color='r', fontsize = 20)
ax.axvspan(pd.to_datetime('2018-08-13'), pd.to_datetime('2018-08-18'), facecolor='orange', alpha=0.7)


##denormalize
predicted = predicted.astype(float)
soms = np.array(soms.mval1)

d_norm = (predicted) * (max(soms) - min(soms)) + min(soms)


d = {'ts' : t, 'rain':rainfall.r15m, 'predicted_soms': predicted}