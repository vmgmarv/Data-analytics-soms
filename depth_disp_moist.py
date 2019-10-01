# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:39:26 2017
"""
import os
import sys
os.path.abspath('/home/marvin/Documents/DYNASLOPE/DYNASLOPE/mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('/home/marvin/Documents/DYNASLOPE/DYNASLOPE/updews-pycodes/Analysis')) #GroundDataAlertLib - l
import pandas as pd
import querySenslopeDb as q
import matplotlib.pyplot as plt
import data_mine_disp as dp
import rtwindow as rtw
import ColumnPlotter as plter
import genproc as g
from datetime import timedelta
from datetime import datetime
 

#col = q.GetSensorList('imusc')
#end = pd.to_datetime('2016-09-05')
#window, config = rtw.getwindow(end)


#start = 100
#column_fix = 'bottom'
#window.start = window.end - timedelta(int(start))

#window.offsetstart = window.start - timedelta(days=(config.io.num_roll_window_ops*window.numpts-1)/48.)
#monitoring = g.genproc(col[0], window, config,column_fix)
#plter.main(monitoring, window, config, plotvel= False)

##################################(Moist)
end_time = '2016-12-25'
sen = 'imuscm'
start_time = '2016-08-25'
    
data = q.GetSomsData(sen, start_time, end_time)
data['ts']=data['ts'].dt.round('1440min')
x = data[data.msgid == 113]
x.drop(['mval2', 'msgid'], axis=1, inplace=True)

dfg = x.groupby ('ts')

def avets(dfg):
    new = dfg.groupby('id')['mval1'].mean()
    return new

ave = dfg.apply(avets)

aveid = pd.DataFrame(ave)
aveid.reset_index(inplace = True)
ave_id = aveid.groupby('ts')


fig = plt.figure()
ax = fig.add_subplot(121)
ax = plter.nonrepeat_colors(ax,len(x.ts.unique()),'plasma')
ax.invert_yaxis()
ax.set_title('SOMS')
plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)

def plotter(dfg,ax):
       ax.plot(dfg.mval1,dfg.id)
       return
    
dfg.apply(plotter,ax)
#####################################(Disp)
end2 = '2016-09-05'
start2 = '2016-09-04'
col2 = 'laysb'
data2 = dp.disp(end2, col2, start2)

data2['ts']=data2['ts'].dt.round('14400min') 

data2.drop(['depth', 'xy', 'vel_xz','vel_xy'], axis=1, inplace=True)
data2['xz'].abs()
dfg2 = data2.groupby ('ts')


#ave_id2= dfg2.apply(avexz)


ax2 = fig.add_subplot(122)
ax2 = plter.nonrepeat_colors(ax2,len(data2.ts.unique()),'plasma')
ax2.invert_xaxis()
ax2.invert_yaxis()
ax2.set_title('Disp')
plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)

def plotter2(dfg2,ax2):
       ax2.plot(dfg2.xz,dfg2.id)
       return
    
dfg2.apply(plotter2,ax2)
