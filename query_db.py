# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 11:20:28 2018
"""

import os
import sys
os.path.abspath('C:\Users\\meryl\\Desktop\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\meryl\\Documents\updews-pycodes-master\Analysis')) #GroundDataAlertLib - line 213
import pandas as pd
import querySenslopeDb as q
import rtwindow as rtw
from datetime import timedelta
import genproc as g

def soms_data(start_date, end_date, site):
    
    db = q.GetSomsData(siteid = site, fromTime = start_date, toTime = end_date)
    df = db[db.msgid == 110]
    df.drop(['msgid','mval2'], axis = 1, inplace = True)
    
    return df

def rainfall_data(start, end, gauge):
    import mysql.connector as sql

    db_connection = sql.connect(host='127.0.0.1', database='senslopedb', user='root', password='alienware091394')
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT * FROM senslopedb.{}".format(gauge))
    
    rainfall = pd.read_sql("SELECT * FROM senslopedb.{}".format(gauge), con=db_connection)
    rainfall = rainfall[(rainfall['timestamp'] >= start) & (rainfall['timestamp'] <= end)]
    rainfall.sort_values(['timestamp'],inplace = True)
    rainfall['timestamp']=rainfall['timestamp'].dt.round('30min')
    rainfall.drop(['name', 'batv1', 'batv2', 'cur', 'boostv1', 'boostv2', 'charge','csq', 'temp', 'hum', 'flashp'],axis=1, inplace =True)
    rainfall ['norm'] = (rainfall['r15m'] - rainfall['r15m'].min()) / (rainfall['r15m'].max() - rainfall['r15m'].min())

    return rainfall

def ground_data(start, end, groundmeas):
    import mysql.connector as sql

    db_connection = sql.connect(host='127.0.0.1', database='senslopedb', user='root', password='alienware091394')
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT * FROM senslopedb.{}".format(groundmeas))

    ground = pd.read_sql("SELECT * FROM senslopedb.{}".format(groundmeas), con=db_connection)
    ground = ground[(ground['timestamp'] >= pd.to_datetime(start)) & (ground['timestamp'] <= pd.to_datetime(end))]
    ground.sort_values(['timestamp'],inplace = True)
    ground['timestamp']=ground['timestamp'].dt.round('30min')
    ground.drop(['meas_type', 'observer_name', 'weather', 'reliability'],axis=1, inplace =True)
    
    return ground

def sensor_data(date_end, sensor, date_start):
    
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
    
    window.offsetstart = window.start - timedelta(days=(config.io.num_roll_window_ops*window.numpts-1)/48.)

    column_fix = 'bottom' 
    config.io.column_fix = column_fix 

    monitoring = g.genproc(col[0], window, config, config.io.column_fix, comp_vel = True)
    monitoring_vel = monitoring.disp_vel.reset_index()[['ts', 'id', 'depth', 'xz', 'xy', 'vel_xz', 'vel_xy']]
    monitoring_vel.sort_values(['ts','id'],inplace = True)
    
    return monitoring_vel
#if __name__ == '__main__':
#    
#    start = '2017-01-01'
#    end = '2017-12-30'
#    site = 'laysam'
#    
#    df = soms_data(start,end,site)
    