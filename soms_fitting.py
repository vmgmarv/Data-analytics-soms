#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:56:33 2018
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

sen = 'lpasam'
start_time = '2017-01-02'
end_time = '2017-01-24'
#data = q.GetSomsData(sen, start_time, end_time)


data = q.GetSomsData(sen, start_time, end_time)

da = rg.regression2(data)
#da['vol'] = (da['grav'] * 1.3)
da['ts']=da['ts'].dt.round('30min')
x = da[da.msgid == 110]
x.drop(['mval2', 'msgid'], axis=1, inplace=True)
    
    #x.loc[(x['id'] >= 5), 'vol'] = 0.465651
#    data.loc[(x['id'] >= 5), 'vol'] = 0.345373
#    data.loc[(data['vol'] >= 0.345373), 'vol'] = 0.345373
#    m = data.ix[(data['nid'] >= 5)]
###########################################################################################################
#dfg = x.groupby (['ts','id'])['vol'].mean()
#    
#da = pd.DataFrame(dfg)
#da = da.reset_index()
#da.sort_values(by=['id'])
#    #dfg2 = da.groupby('ts')
    
dfg2= x.groupby('ts')
    

    
fig = plt.figure()
ax = fig.add_subplot(111)
ax = plter.nonrepeat_colors(ax,len(data.ts.unique()),'plasma')
ax.invert_yaxis()
ax.set_title('SOMS in {} \n From  {} to {}'.format(sen,start_time,end_time), fontsize=22)
    #ax.set_xlim([0,0.50])
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.5)
plt.ylabel('Depth [m]',fontsize=20)
    #pl.xlabel(r'$\psi$ [m]',fontsize=20)
    #pl.subplot(122)
plt.xlabel(r'$\theta[cm^3/cm^3]$',fontsize=20)
    
def plot(dfg2):
    dfg2.sort_values(by=['mval1'])
    #new = dfg.groupby('id')['mval1']
    ax.plot(abs(dfg2.vol),dfg2.id)
    return
    
ave = dfg2.apply(plot)
