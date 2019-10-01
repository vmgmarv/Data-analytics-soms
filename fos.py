# -*- coding: utf-8 -*-
"""
Created on Mon Sep 03 08:58:22 2018
"""
############
import os
import sys
os.path.abspath('C:\Users\\meryl\\Documents\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\meryl\\Documents\updews-pycodes\Analysis')) #GroundDataAlertLib - line 21
import math
import regression as rg
import pandas as pd
import matplotlib.pyplot as plt

start = '2017-09-08'
end = '2017-09-17'
site = 'laysam'
gauge = 'laysaw'
site_code = 'lay'

import mysql.connector as sql

db_connection = sql.connect(host='192.168.150.129', database='senslopedb', user='dyna_staff', password='accelerometer')

def soms_calibration (start_time, end_time, site_soms):
    df = pd.read_sql("SELECT * FROM senslopedb.{}".format(site_soms), con=db_connection)
    df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

    da = rg.regression_lay(df)

    da['timestamp']=da['timestamp'].dt.round('30min')
    x = da[da.msgid == 110]
    x.drop(['mval2', 'msgid'], axis=1, inplace=True)

    def roll_ave(dfg):
        dfg['vol_roll_mean'] = dfg['vol'].rolling(7).mean()
        return dfg
    soms = x.groupby('id').apply(roll_ave)
    ave = soms.vol.mean(0)
    return soms,ave
#########################################################################################################r

###########################################################################################################rainfall
def rainfall (start_time, end_time, gauge):
    df = pd.read_sql("SELECT * FROM senslopedb.{}".format(gauge), con=db_connection)
    df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    df.sort_values(['timestamp'],inplace = True)
    df['timestamp']=df['timestamp'].dt.round('30min')
    df.drop(['name', 'batv1', 'batv2', 'cur', 'boostv1', 'boostv2', 'charge','csq', 'temp', 'hum', 'flashp'],axis=1, inplace =True)
    df['norm'] = (df['r15m'] - df['r15m'].min()) / (df['r15m'].max() - df['r15m'].min())
    df ['roll_sum_1d'] = df['r15m'].rolling(48).sum()
    df ['roll_sum_3d'] = df['r15m'].rolling(144).sum()
    rainfall = df.reset_index(drop=True)
    
    return rainfall

rain = rainfall(start, end, gauge)
############################################################################################################

df,average = soms_calibration(start,end,site)
df = df[(df['id'] >1 ) & (df['id'] < 7)]

dg = df.groupby('timestamp')

##input variables
# Yw, (Yn = Ys - Yd), q, Sw, H, h, Cs, Cr,theta, phi
Yw = 9.8     ###unit weight of water (kN/m^3)
Ys = 18.0 ###saturated unit soil weight (kN/m^3)
Yd = 16.0 ###dry unit soil weight (kN/m^3)
Yn = Ys - Yd ###unit weight of available water in the soil layer
q = 0.0 ###additional load on the soil surface (kN/m^2)
#Sw = 50.0 ### degree of soil saturation (cm^3/cm^3)
H = 5.0 ### total depth of the soil above the failure plane (m)
h = 3.0 ### saturated thickness of the soil above the failure plane (m)
Cs = 25.0 ### effective soil cohesion (kN/m^2)
Cr = 0.0 ### effective root cohesion (kN/m^2)
theta = 30.0 * (3.14159265359 / 180) ### slope angle
phi = 38 * (3.14159265359 / 180) ### angle of internal friction of the soil
# m = moisture content
Gs = 2.80 #specific gravity 2.70 for inorganic clay
n = 0.37 # percent porosity
e = n / (1 - n) #void ratio

def ave(dg):
    dg['vol_ave'] = dg['vol'].mean()
    dg['Sw'] = (dg.vol_ave / n) * 100
    A = Cs + Cr
    B = math.sin(theta) * (dg.Sw * Yn * (H - h) + q * math.cos(theta) + Yd * H  + Yn * h)
    C = math.tan(phi) / math.tan(theta)
    D = Yw * ((h / H) + dg.Sw * (1 - (h / H)))
    E = dg.Sw * Yn * (1 - (h / H) + (q * math.cos(theta) / H) + Yd + (h / H) * Yn)
    dg['FS'] = (A / B) + C * (1 - (D / E))
    return dg

df = dg.apply(ave)

des = df.FS.describe()
##########################################################################plotting

fig, ax1 = plt.subplots()
ax1.plot(df.timestamp, df.FS, color = 'blue')
ax1.set_xlabel('timestamp')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('FS',fontsize=20, color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(df.timestamp, df.vol_ave, color = 'red', alpha = 0.5)
ax2.set_ylabel(r'Soms$[cm^3 / cm^3]$', color='r', fontsize=20)
ax2.tick_params('y', colors='r')
plt.title('FS vs SOMS', fontsize = 30)


print des 
#######################################################################################################################
#axnew = fig.add_axes([0.2, 0.9, 0.85, 0.2])