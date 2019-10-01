# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:33:52 2019
"""

import os
import sys
os.path.abspath('C:\Users\\meryl\\Desktop\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\meryl\\Documents\updews-pycodes-master\Analysis')) #GroundDataAlertLib - line 21
import matplotlib.pyplot as plt
import pandas as pd
import data_mine_disp as dp
import numpy as np
import ColumnPlotter as plter
from matplotlib import colors as mcolors
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import mysql.connector as sql



start = '2018-01-01'
end = '2018-03-30'
sensor = 'tilt_marta'
seg_len = 1.5



db_connection = sql.connect(host='192.168.150.75', database='senslopedb', user='pysys_local', password='NaCAhztBgYZ3HwTkvHwwGVtJn5sVMFgg')


def data(start, end, sensor):
    read = db_connection.cursor()
    query = "SELECT * FROM senslopedb.%s" %(sensor)
    query += " WHERE ts BETWEEN '%s' AND '%s'" %(start, end)
      
    
    read.execute(query)
    d = pd.DataFrame(read.fetchall())
    d.columns = read.column_names

    d.drop(['batt'], axis=1, inplace=True)
    d = d[d.type_num == 11]
    return d


def accel_to_lin_xz_xy(data, seg_len):

    #DESCRIPTION
    #converts accelerometer data (xa,ya,za) to corresponding tilt expressed as
    #horizontal linear displacements values, (xz, xy)
    
    #INPUTS
    #seg_len; float; length of individual column segment
    #xa,ya,za; array of integers; accelerometer data (ideally, -1024 to 1024)
    
    #OUTPUTS
    #xz, xy; array of floats; horizontal linear displacements along the planes 
    #defined by xa-za and xa-ya, respectively; units similar to seg_len
    
    xa = data.xval.values
    ya = data.yval.values
    za = data.zval.values

    theta_xz = np.arctan(za / (np.sqrt(xa**2 + ya**2)))
    theta_xy = np.arctan(ya / (np.sqrt(xa**2 + za**2)))
    xz = seg_len * np.sin(theta_xz)
    xy = seg_len * np.sin(theta_xy)
    
    data['xz'] = np.round(xz,4)
    data['xy'] = np.round(xy,4)
    
    return data

data = data(start, end, sensor)
filtered = filt.apply_filters(data)
data = accel_to_lin_xz_xy(filtered, seg_len)
    
n = data.node_id.max() + 1 

cols = np.arange(1,n)

x = data[data.node_id == 5]

plt.plot(x.ts,x.xz,linewidth  = 2, marker = 'o', label = 'Node {}'.format(x))




#for i in cols:
#    plt.subplot(len(cols), 1, i, label = '{}'.format(i))
#    plt.legend('{}'.format(i),loc=2)
#    x = data[data.node_id == i]
#    plt.plot(x.ts,x.xz)
#    i += 1
#plt.show()

#fig, (axes) = plt.subplots(len(cols), 1, sharex = True)
#fig.subplots_adjust(hspace = 0)
#
#for i in cols:
#    j = i - 1
#    print j
#    x = data[data.node_id == i]
#    axes[j].plot(x.ts,x.xz, label = '{}'.format(i))
#    plt.legend('{}'.format(i),loc=2)
