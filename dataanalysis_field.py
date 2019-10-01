#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 08:58:57 2018
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

csv = pd.read_csv('/home/marvin/Documents/DYNASLOPE/DYNASLOPE/experiment/hinsb.csv')
da = pd.DataFrame(csv)
df = da[da.msgid == 110]
data = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
####################################################################################################################
#xx = pd.DataFrame(x)
#xx.columns.duplicated()
#g = xx.loc[:,[False,True]]
l = data['ts'].astype(str)
d = pd.to_datetime(l)
data.drop(['ts'], axis=1, inplace=True)
data ['ts'] = pd.to_datetime(l)
###################################################################################################################
dfg = data.groupby ('ts')

fig = plt.figure()
ax = fig.add_subplot(111)
ax = plter.nonrepeat_colors(ax,len(data.ts.unique()),'plasma')
ax.invert_yaxis()
ax.set_title('Experimental (HINSB)')
plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
plt.ylabel('Depth [m]',fontsize=18)
    #pl.xlabel(r'$\psi$ [m]',fontsize=20)
    #pl.subplot(122)
plt.xlabel(r'$\theta$',fontsize=18)

def plot(dfg):
    ax.plot(dfg.x,dfg.nid)
    return
    
ave = dfg.apply(plot)
#####################################################33
#fig, (axes) = plt.subplots(9, 1, sharex = True, sharey= False)
#fig.subplots_adjust(hspace=0)
#    
#plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
#    
#    
#for i in range (1, 9):
#    x = data[data.nid == i]
#    
#    axes [i].plot(x.ts, x.x, color = 'orange')
#    axes[i].yaxis.set_visible(False)
#
