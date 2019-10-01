# -*- coding: utf-8 -*-
"""
Created on Mon Oct 01 15:13:45 2018
"""

import os
import sys
os.path.abspath('C:\Users\\meryl\\Desktop\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\meryl\\Documents\updews-pycodes-master\Analysis')) #GroundDataAlertLib - line 21
import querySenslopeDb as q
import matplotlib.pyplot as plt
import pandas as pd
import mysql.connector as sql

db_connection = sql.connect(host='192.168.150.129', database='senslopedb', user='dyna_staff', password='accelerometer')

def rainfall (start_time, end_time, gauge):
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

start = '2018-01-15'
end = '2018-10-10'
site = 'blcsbm'
gauge = 'nagtbw'



df = q.GetSomsData(site, start, end)
#df['roll_sum_1d'] = df['mval1'].rolling(48).mean()
#df = df[df.msgid == 21]
#############################################################################Rainfall
rainfall = rainfall(start, end, gauge) 
rainfall = rainfall.dropna()
##############################################################################

df['ts']=df['ts'].dt.round('30min')

n = int(df.id.max()) + 1

fig, (axes) = plt.subplots(n, 1, sharex = True, sharey= False)
fig.subplots_adjust(hspace=0)
fig.suptitle('Data points to be used for ANN ({} to {})'.format(start,end), fontsize = 20)

plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
plt.xticks(rotation=45)
axes[0].plot(rainfall.timestamp, rainfall.roll_sum_1d, color = 'blue')
axes[0].yaxis.set_visible(False)
for i in range (1, n):
    print i
    x = df[df.id == i]
    
    axes[i].plot(x.ts, x.roll_sum_1d, color = 'orange')
    axes[i].yaxis.set_visible(False)
    

for ax in axes[0:n]:
    ax.axvspan(pd.to_datetime('2018-08-13'), pd.to_datetime('2018-08-18'), facecolor='red', alpha=0.7)
    ax.axvspan(pd.to_datetime('2018-08-25'), pd.to_datetime('2018-08-26'), facecolor='red', alpha=0.7)
    ax.axvspan(pd.to_datetime('2018-08-11'), pd.to_datetime('2018-08-16'), facecolor='yellow', alpha=0.7)
    ax.axvspan(pd.to_datetime('2018-08-24'), pd.to_datetime('2018-08-29'), facecolor='yellow', alpha=0.7)













#dfg2= x.groupby('ts')
#
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1 = plter.nonrepeat_colors(ax1,500,'plasma')
#ax1.invert_yaxis()
#ax1.set_title('Realtime (HINSB)', fontsize=22)
#plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
#plt.ylabel('Depth [m]',fontsize=18)
#    #pl.xlabel(r'$\psi$ [m]',fontsize=20)
#    #pl.subplot(122)
#plt.xlabel('mval1',fontsize=18)
#
#def plot(dfg):
#    ax1.plot(dfg.mval1,dfg.id)
#    return
#    
#ave = dfg2.apply(plot)
###########################################################################################################################################
###########################################################################################################################################
