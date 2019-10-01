# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 13:49:48 2017
"""
#import pandas as pd
import os
import sys
os.path.abspath('C:\Users\\Vidal Marvin Gabriel\\Desktop\DYNASLOPE\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\Vidal Marvin Gabriel\\Desktop\DYNASLOPE\updews-pycodes\Analysis')) #GroundDataAlertLib - line 21
import querySenslopeDb as q
import matplotlib.pyplot as plt
import pandas as pd
import data_mine_disp as dp

n = 7
##################################(Moist)
end = '2016-09-05'
col = 'imuscm'
start = '2016-07-21'
    
data = q.GetSomsData(col, start, end)
data['ts']=data['ts'].dt.round('30min')
x = data[data.msgid == 113]

x_x = pd.DataFrame(x)
x_x.drop('mval2', 1)
#x = data[data.index %2 != 0].set_index('ts') #sets only odd numbers
#data[data.index %2 == 0] #sets only even numbers
#data[data.index %2 == 0].reset_index()
#data[data.index %2 == 0].set_index('ts') #sets index into timestamp
nod = x_x[x_x.id == n]
nod ['moist_rate'] = nod['mval1'] - nod['mval1'].shift()
#nod ['rate'] = np.where((nod['mval1'] - nod['mval1'].shift()) >=10, (nod['mval1'] - nod['mval1'].shift()),0) 

rate = nod ['moist_rate'].abs()

moist_rate = pd.DataFrame(rate) #putting it into dataframe
result = pd.concat([x_x,moist_rate],axis = 1, join ='inner')
result.set_index('ts',inplace = True)

#result.moist_rate = result.moist_rate.fillna(0)
#result.plot(x='ts', y='moist_rate')
#####################################(Disp)
end2 = '2016-09-05'
start2 = 44
col2 = 'imusc'
data2 = dp.disp(end2, col2, start2)
data2['ts']=data2['ts'].dt.round('30min')
x_d = data2.set_index('ts')

nod_d = x_d[x_d.id == n]

nod_d['disp_rate'] = (nod_d['xz'] - nod_d['xz'].values[0]) * 1000 #first pos only
#nod_d ['disp_rate'] = nod_d['xz'] - nod_d['xz'].shift() * 1000
#nod_d['rate'] = np.where((nod_d['xz'] - nod_d['xz'].shift())>=0.1,(nod_d['xz'] - nod_d['xz'].shift()),0)           
rate2 = nod_d['disp_rate'].abs()

result2 = pd.DataFrame(rate2)

#fr = pd.concat([result, result2], axis=1)

final = result
final['disp_rate'] = result2
final.drop(['mval2'],axis =1)
        
final = final.dropna(thresh=1)
#final.plot(x = 'ts', y ='moist_rate')
#final.plot(x = 'ts', y ='disp_rate')
########################################(plotting)
#fig1 = plt.figure()
#plt.plot(rate, label='Moisture')
#plt.xlabel('ts')
#plt.plot(rate2, label='Displacement')
#plt.axhline(y = 50, xmin=0, xmax=1, c = 'red', hold=None)
#plt.xlabel('ts')
#plt.show()
######################################(combining reulst 1 and result 2)
#result = pd.concat([result1, result2], axis=1) 
#result.moist_rate = result.moist_rate.fillna(0) #making nan values to zero
#result.disp_rate = result.disp_rate.fillna(0)

fig1 = plt.figure()
plt.plot(final['moist_rate'], label = 'Moisture')
plt.plot(final['disp_rate'], label = 'Displacement')
#plt.axhline(y = 0.5 * 1000, xmin=0, xmax=1, c = 'red', hold=None)
plt.title('Node 7')
plt.legend()
#plt.axhline(y = 50, xmin=0, xmax=1, c = 'red', hold=None)
#plt.xlabel('ts')

