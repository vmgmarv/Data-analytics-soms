# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:53:36 2017
"""

import os
import sys
os.path.abspath('C:\Users\\Vidal Marvin Gabriel\\Desktop\DYNASLOPE\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\Vidal Marvin Gabriel\\Desktop\DYNASLOPE\updews-pycodes\Analysis')) #GroundDataAlertLib - line 21
import pandas as pd
import querySenslopeDb as q
import rtwindow as rtw
import genproc as g
from datetime import timedelta


def somsdata (end_time, sen, start_time):
    data = q.GetSomsData(sen, start_time, end_time)
    #data['ts']=data['ts'].dt.round('30min')
    x = data[data.msgid == 110]
    x.drop(['mval2'], axis=1, inplace=True)
    
    return x


def disp(date_end, sensor, date_start):
    #str.....sila lahat
    end = pd.to_datetime(date_end) #inputs specified time
    col = q.GetSensorList(sensor) #inputs the name of the sensor
    start = (date_start) #inputs monitoring window
    window, config = rtw.getwindow(end)
    window.start = pd.to_datetime(start) 
    while True:
                start = date_start
                try:
                    window.start = window.end - timedelta(int(start))
                    break
                except:
                    try:
                        window.start = pd.to_datetime(start)
                        break
                    except:
                        print 'datetime format or integer only'
                        continue
    
    window.offsetstart = window.start - timedelta(days=(config.io.num_roll_window_ops*window.numpts-1)/48.) #fixes the time (offsets) for the 3 day monitoring

#somsdata = q.GetSomsData(sensor, window.offsetstart, end)                                        
    column_fix = 'bottom' #i dont know the use
    config.io.column_fix = column_fix #i dont know the use
    
#getdispdata = q.GetRawAccelData #i dont know yet!!!!!!!!!!!!!!!!
    monitoring = g.genproc(col[0], window, config, config.io.column_fix, comp_vel = True)
    monitoring_vel = monitoring.disp_vel.reset_index()[['ts', 'id', 'depth', 'xz', 'xy', 'vel_xz', 'vel_xy']] #ColumnPlotter.py line 597
    monitoring_vel.sort_values(['ts','id'],inplace = True) #sorts values ts and id in plance not random!
    #monitoring_vel = monitoring_vel.sort_values(['ts','id'],inplace = True)same as  monitoring_vel.sort_values(['ts','id'],inplace
    
    #monitoring_vel.to_csv("{} {} to {}.csv".format(col[0].name,end.strftime('%Y-%m-%d_%H-%M'),window.start.strftime('%Y-%m-%d_%H-%M')))#save the data in csv file
    
    return monitoring_vel
