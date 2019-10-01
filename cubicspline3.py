#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:29:26 2018
"""
import os
import sys
os.path.abspath('C:\Users\\meryl\\Documents\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\meryl\\Documents\updews-pycodes\Analysis')) #GroundDataAlertLib - line 21
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy.polynomial.polynomial as poly
import ColumnPlotter as plter
from matplotlib import rc

csv = pd.read_csv('C:\Users\\meryl\\Documents\experiment\hinsb.csv')
dc = pd.DataFrame(csv)
df = dc[dc.msgid == 110]
data = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)

l = data['ts'].astype(str)
d = pd.to_datetime(l)
data.drop(['ts'], axis=1, inplace=True)
data ['ts'] = pd.to_datetime(l)
data.loc[(data['x'] > 1181), 'x'] = 1181
data.loc[(data['x'] < 896), 'x'] = 897

csv2 = pd.read_csv('C:\Users\\meryl\\Documents\experiment\laysa.csv')
dc2 = pd.DataFrame(csv)
df2 = dc[dc.msgid == 110]
data2 = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)

l2 = data['ts'].astype(str)
d2 = pd.to_datetime(l)
data2.drop(['ts'], axis=1, inplace=True)
data2 ['ts'] = pd.to_datetime(l)
data2.loc[(data2['x'] > 1185), 'x'] = 1185
data2.loc[(data2['x'] < 980), 'x'] = 980
#################################################################################################
d0 = pd.DataFrame(np.random.randint(1,123,size=(100)), columns = ['A'])
d0['A'] = d0['A'] / 1000
d0['B'] = pd.DataFrame(np.random.randint(717,897,size=(1000)))

d1 = pd.DataFrame(np.random.randint(123,189,size=(100)), columns = ['A'])
d1['A'] = d1['A'] / 1000
d1['B'] = pd.DataFrame(np.random.randint(897,951,size=(100)))

d2 = pd.DataFrame(np.random.randint(189,390,size=(100)), columns = ['A'])
d2['A'] = d2['A'] / 1000
d2['B'] = pd.DataFrame(np.random.randint(951,1027,size=(100)))

d3 = pd.DataFrame(np.random.randint(390,479,size=(100)), columns = ['A'])
d3['A'] = d3['A'] / 1000
d3['B'] = pd.DataFrame(np.random.randint(1027,1041,size=(100)))

d4 = pd.DataFrame(np.random.randint(479,500,size=(100)), columns = ['A'])
d4['A'] = d4['A'] / 1000
d4['B'] = pd.DataFrame(np.random.randint(1041,1124,size=(100)))

d5 = pd.DataFrame(np.random.randint(500,539,size=(100)), columns = ['A'])
d5['A'] = d5['A'] / 1000
d5['B'] = pd.DataFrame(np.random.randint(1124,1154,size=(100)))

d6 = pd.DataFrame(np.random.randint(539,634,size=(100)), columns = ['A'])
d6['A'] = d6['A'] / 1000
d6['B'] = pd.DataFrame(np.random.randint(1154,1181,size=(100)))

d7 = pd.DataFrame(np.random.randint(626,634,size=(100)), columns = ['A'])
d7['A'] = d7['A'] / 1000
d7['B'] = pd.DataFrame(np.random.randint(1175,1181,size=(100)))

d8 = pd.DataFrame(np.random.randint(626,1000,size=(100)), columns = ['A'])
d8['A'] = d8['A'] / 1000
d8['B'] = pd.DataFrame(np.random.randint(1181,1217,size=(100)))

frames = [d1, d2, d3, d4, d5, d6, d7, d8]

result = pd.concat(frames)

data_x = result['B']
data_y = result['A']/ 1.48

x = np.array(data_x)
y = np.array(data_y)

xp = np.linspace(x.min(),x.max(),100)

a,b,c,d = np.polyfit(x, y, 3)
coefs = poly.polyfit(x, y, 3)
ffit = poly.polyval(xp, coefs)

xp = np.linspace(x.min(),x.max(),100)

z = np.polyfit(x, y, 3)

p = np.poly1d(z)

p30 = np.poly1d(np.polyfit(x, y, 30))
x_new = np.linspace(x[0], x[-1], 50)



data['grav'] = p(data['x'])
data['vol'] = data['grav'] / 1.21 
data.loc[(data['nid'] >= 5), 'vol'] = 0.345373
data.loc[(data['vol'] >= 0.345373), 'vol'] = 0.345373
m = data.ix[(data['nid'] >= 5)]
#dfg = data.groupby ('ts')

datafirst = data.loc[0:47, 'nid':'vol']

#################################################################################################
d02 = pd.DataFrame(np.random.randint(1,98,size=(100)), columns = ['A'])
d02['A'] = d02['A'] / 1000
d02['B'] = pd.DataFrame(np.random.randint(699,980,size=(1000)))

d12 = pd.DataFrame(np.random.randint(98,160,size=(100)), columns = ['A'])
d12['A'] = d12['A'] / 1000
d12['B'] = pd.DataFrame(np.random.randint(980,1030,size=(1000)))

d22 = pd.DataFrame(np.random.randint(161,240,size=(200)), columns = ['A'])
d22['A'] = d22['A'] / 1000
d22['B'] = pd.DataFrame(np.random.randint(1031,1062,size=(1000)))

d32 = pd.DataFrame(np.random.randint(241,330,size=(100)), columns = ['A'])
d32['A'] = d32['A'] / 1000
d32['B'] = pd.DataFrame(np.random.randint(1063,1142,size=(100)))

d42 = pd.DataFrame(np.random.randint(331,480,size=(100)), columns = ['A'])
d42['A'] = d42['A'] / 1000
d42['B'] = pd.DataFrame(np.random.randint(1143,1184,size=(100)))

d52 = pd.DataFrame(np.random.randint(481,520,size=(100)), columns = ['A'])
d52['A'] = d52['A'] / 1000
d52['B'] = pd.DataFrame(np.random.randint(1182,1184,size=(100)))

d62 = pd.DataFrame(np.random.randint(521,580,size=(100)), columns = ['A'])
d62['A'] = d62['A'] / 1000
d62['B'] = pd.DataFrame(np.random.randint(1185,1194,size=(100)))

d72 = pd.DataFrame(np.random.randint(581,1000,size=(100)), columns = ['A'])
d72['A'] = d72['A'] / 1000
d72['B'] = pd.DataFrame(np.random.randint(1196,1217,size=(100)))


frames2 = [d02, d12, d22, d32, d42, d52, d62, d72]

result2 = pd.concat(frames2)

data_x2 = result2['B']
data_y2 = result2['A']/ 1.48

x2 = np.array(data_x2)
y2 = np.array(data_y2)

xp2 = np.linspace(x2.min(),x2.max(),100)

a2,b2,c2,d2 = np.polyfit(x2, y2, 3)
coefs2 = poly.polyfit(x2, y2, 3)
ffit2 = poly.polyval(xp2, coefs2)

xp2 = np.linspace(x2.min(),x2.max(),100)

z2 = np.polyfit(x2, y2, 3)

p2 = np.poly1d(z2)

p30_2 = np.poly1d(np.polyfit(x2, y2, 30))
x_new2 = np.linspace(x2[0], x2[-1], 50)



data2['grav'] = p2(data2['x'])
data2['vol'] = data2['grav'] / 1.21 
data2.loc[(data2['nid'] >= 4), 'vol'] = 0.465651
data2.loc[(data2['vol'] >= 0.465651), 'vol'] = 0.465651

m2 = data2.ix[(data2['nid'] >= 4)]

#dfg2 = data2.groupby ('ts')

datafirst2 = data2.loc[0:47, 'nid':'vol']

##################################################################################################
fig, (axes) = plt.subplots(1, 2, sharex = True, sharey= True)
fig.patch.set_facecolor('#AAB8CB')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
fig.subplots_adjust(wspace=0.06)

#fig.suptitle('Regression analysis', fontsize=30)

fig.text(0.5, 0.04, 'Sensor response [$mV$]', fontsize= 25, ha='center')
fig.text(0.07, 0.5, r'Water content [$\%vol$]',fontsize= 25, va='center', rotation='vertical')

############################################################################################################
axes[0].scatter(x, y, color='black', label = 'Data points')
axes[0].plot(xp, ffit, label = 'Spline', linewidth=4, color = 'orangered')
axes[0].grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.5)

axes[0].set_title('Brgy.1&2, Hinabangan (Silt)', fontsize = 25)

axes[0].tick_params(axis='both', which = 'major',labelsize = 20)
#plt.title('REGRESSION - Laygayon(Clay sample)',fontsize = 30)

#plt.ylabel(r'$\theta [\%grams/grams]$',fontsize=20, fontweight = 'bold')

##################################################################################################

axes[1].scatter(x2, y2, color = 'black', label = 'Data points')
axes[1].plot(xp2, ffit2, label = 'Spline', linewidth=4, color = 'orangered')
axes[1].grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.5)

#plt.title('REGRESSION - Laygayon(Clay sample)',fontsize = 30)
axes[1].set_title('Brgy. Laygayon (Clay)', fontsize = 25)
    
axes[1].tick_params(axis='both', which = 'major',labelsize = 20)
leg = axes[1].legend(loc='upper left', bbox_to_anchor=(0.01, 1), ncol=1, framealpha = 1, frameon = True, borderpad = 1,
          shadow=True, fancybox=True, fontsize=15)

#plt.ylabel(r'$\theta [\%grams/grams]$',fontsize=20, fontweight = 'bold')
#leg = axes[1].legend(loc='best', ncol=1, shadow=True, fancybox=True, fontsize=20)
#leg.get_frame().set_alpha(0.5)
####################################################################################################################
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
#rc('axes', linewidth=2)
#fig2, (axes2) = plt.subplots(1, 2, sharex = True, sharey= True)
#fig2.subplots_adjust(wspace=0)
#rc('axes', linewidth=2)
#fig2.suptitle('Sensor column data', fontsize=30)
#
#fig2.text(0.5, 0.04, 'sensor response',fontsize= 25, fontweight = 'bold', ha='center')
#fig2.text(0.07, 0.5, r'$\theta [\%grams/grams]$',fontsize= 25, fontweight = 'bold', va='center', rotation='vertical')
#plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.5)
####################################################################################################################
#axes2[0].plot(datafirst['vol'], datafirst['nid'], linewidth=3, color = 'orange')
#plt.ylim((1,9))
#axes2[0].invert_yaxis()
#axes2[0].grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.5)
#
#axes2[0].set_title('Brgy.1\&2, Hinabangan', fontsize = 25)
#
#axes2[0].tick_params(axis='both', which = 'major',labelsize = 20)
#####################################################################################################################
#axes2[1].plot(datafirst2['vol'], datafirst2['nid'], linewidth=3, color = 'orange')
#axes2[1].grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.5)
#
#axes2[1].set_title('Brgy.Laygayon', fontsize = 25)
#
#axes2[1].tick_params(axis='both', which = 'major',labelsize = 20)
#####################################################################################################################
fig2 = plt.figure()
fig2.patch.set_facecolor('#AAB8CB')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

#plt.title('Sensor columns', fontsize = 30)
fig2.text(0.5, 0.04, r'Water content [$\%vol$]',fontsize= 25, ha='center')
fig2.text(0.07, 0.5, r'Depth $[m]$',fontsize= 25, va='center', rotation='vertical')
plt.plot(datafirst['vol'], datafirst['nid'], label = 'Brgy.1&2, Hinabangan', linewidth=3.5, color = 'darkorange')
plt.plot(datafirst2['vol'], datafirst2['nid'], label = 'Brgy.Laygayon', linewidth=3.5, color = 'seagreen')
plt.tick_params(axis='both', which = 'major',labelsize = 20)

plt.ylim((1,9))
plt.gca().invert_yaxis()
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.5)

leg2 = plt.legend(loc='best', bbox_to_anchor=(1.1, 1.05), ncol=1, framealpha = 1, frameon = True, borderpad = 1,
          shadow=True, fancybox=True, fontsize=20)
