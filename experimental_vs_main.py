#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:48:09 2018
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(1,os.path.abspath('/home/marvin/Documents/DYNASLOPE/DYNASLOPE/updews-pycodes/Analysis')) #GroundDataAlertLib - line 21
sys.path.insert(1,os.path.abspath('/home/marvin/Documents/DYNASLOPE/DYNASLOPE/updews-pycodes/Analysis')) #GroundDataAlertLib - line 21
import ColumnPlotter as plter
from fractions import Fraction
import querySenslopeDb as q


sen = 'hinsbm'
start_time = '2018-04-18'
end_time = '2018-04-19'
#######################################################################################################
csv = pd.read_csv('/home/marvin/Documents/DYNASLOPE/DYNASLOPE/experiment/hinsb.csv')
da = pd.DataFrame(csv)
df = da[da.msgid == 110]
data = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
#######################################################################################################

l = data['ts'].astype(str)
d = pd.to_datetime(l)
data.drop(['ts'], axis=1, inplace=True)
data ['ts'] = pd.to_datetime(l)

dfg = data.groupby ('ts')
#########################################################################################################################################
#########################################################################################################################################
df = q.GetSomsData(sen, start_time, end_time)
#da['ts']=da['ts'].dt.round('30min')
x = df[df.msgid == 110]
x.drop(['mval2', 'msgid'], axis=1, inplace=True)
    
dfg2= x.groupby('ts')

#f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
#ax1.plot(x, y)
#ax1.set_title('Sharing Y axis')
#ax2.scatter(x, y)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1 = plter.nonrepeat_colors(ax1,len(data.ts.unique()),'plasma')
ax1.invert_yaxis()
ax1.set_title('Realtime (HINSB)', fontsize=22)
plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
plt.ylabel('Depth [m]',fontsize=18)
    #pl.xlabel(r'$\psi$ [m]',fontsize=20)
    #pl.subplot(122)
plt.xlabel('mval1',fontsize=18)

def plot(dfg):
    ax1.plot(dfg.x,dfg.nid)
    return
    
ave = dfg.apply(plot)
###########################################################################################################################################
###########################################################################################################################################
ax2 = fig.add_subplot(122)
ax2 = plter.nonrepeat_colors(ax2,len(data.ts.unique()),'plasma')
ax2.invert_yaxis()
ax2.set_title('SOMS in {} \n From  {} to {}'.format(sen,start_time,end_time), fontsize=22)
plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
plt.ylabel('Depth [m]',fontsize=18)
plt.xlabel('mval1',fontsize=18)
    
def plot(dfg2):
    dfg2.sort_values(by=['id'])
    ax2.plot(abs(dfg2.mval1),dfg2.id)
    return
    
ave2 = dfg2.apply(plot)


