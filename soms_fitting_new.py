# -*- coding: utf-8 -*-
"""
Created on Thu Jul 05 13:48:17 2018
"""

import os
import sys
os.path.abspath('C:\Users\\meryl\\Documents\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\meryl\\Documents\updews-pycodes\Analysis')) #GroundDataAlertLib - line 21
import querySenslopeDb as q
import matplotlib.pyplot as plt
import pandas as pd
import s_d_db as d
import ColumnPlotter as plter
import regression as rg
import numpy as np
from scipy.interpolate import UnivariateSpline
import time
import matplotlib as mpl

startofscript = time.time()
sen = 'laysbm'
start_time = '2017-03-05'
end_time = '2017-03-10'
gauge = 'laysaw'
##########################################################################rainfall
start = '2017-03-01'
end = '2017-03-20'

import mysql.connector as sql

db_connection = sql.connect(host='127.0.0.1', database='senslopedb', user='root', password='senslope')

rainfall = pd.read_sql("SELECT * FROM senslopedb.{}".format(gauge), con=db_connection)
rainfall = rainfall[(rainfall['timestamp'] >= start) & (rainfall['timestamp'] <= end)]
rainfall.sort_values(['timestamp'],inplace = True)
rainfall['timestamp']=rainfall['timestamp'].dt.round('30min')
rainfall.drop(['name', 'batv1', 'batv2', 'cur', 'boostv1', 'boostv2', 'charge','csq', 'temp', 'hum', 'flashp'],axis=1, inplace =True)
rainfall ['norm'] = (rainfall['r15m'] - rainfall['r15m'].min()) / (rainfall['r15m'].max() - rainfall['r15m'].min())
h = rainfall.describe()

rainfall ['roll_sum'] = rainfall['r15m'].rolling(48).sum()
rain = rainfall.fillna(0)
rain.to_csv('rainfall', sep='\t')
###########################################################################################

data = q.GetSomsData(sen, start_time, end_time)

da = rg.regression3(data)

da['ts']=da['ts'].dt.round('30min')
x = da[da.msgid == 110]
x.drop(['mval2', 'msgid'], axis=1, inplace=True)


def roll_ave(dfg):
    dfg['vol_roll_mean'] = dfg['vol'].rolling(7).mean()
    dfg.sort(['id'])
    return dfg
dfg2 = x.groupby('id').apply(roll_ave)

fig = plt.figure()
ax = fig.add_subplot(111)
ax = plter.nonrepeat_colors(ax,len(data.ts.unique()),'plasma')
ax.invert_yaxis()
#ax.set_title('SOMS in {} \n From  {} to {}'.format(sen,start_time,end_time), fontsize=22)
    #ax.set_xlim([0,0.50])
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.5)
plt.ylabel('Depth [m]',fontsize=20)
plt.xlabel(r'$\theta[cm^3/cm^3]$',fontsize=20)
#    
def plot(dfg2):
    ax.plot(abs(dfg2.vol_roll_mean),dfg2.id, marker = 'H')
    return
ave = dfg2.groupby('ts').apply(plot)

ax1 = fig.add_axes([0.09, 0.91, 0.85, 0.07])
ax1.plot(rain.timestamp,rain.roll_sum, color = 'blue', linewidth = 2)
ax1.axvspan(start_time, end_time, facecolor='orange', alpha=0.4)  
plt.ylabel('r24h [mm]')

#ax2 = fig.add_axes([0.92, 0.11, 0.02, 0.77])

#cmap = mpl.cm.plasma
#count = x.describe()
#norm = mpl.colors.Normalize(vmin=0, vmax=int(count['id'].iloc[0]))
#
#cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
#                                   norm=norm,
#                                   orientation='vertical', spacing='proportional')
#cb1.set_label('ts(30)', fontsize = 20, labelpad=-50, y=1.05, rotation=0)
#ax2.tick_params(axis='both', which = 'major',labelsize = 20)

endofscript = time.time()
#
runtime = endofscript - startofscript
print 'runtime = {}'.format(runtime)