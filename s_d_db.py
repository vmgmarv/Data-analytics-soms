# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:12:31 2017
"""

import os
import sys
os.path.abspath('C:\Users\\Vidal Marvin Gabriel\\Desktop\DYNASLOPE\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\Vidal Marvin Gabriel\\Desktop\DYNASLOPE\updews-pycodes\Analysis')) #GroundDataAlertLib - line 21
import soms_disp_db as ddb
import pandas as pd
import rtwindow as rtw
#import soms_col_plot as plotter
import querySenslopeDb as q
import matplotlib.pyplot as plt
import ColumnPlotter as plter

###################################################################################################disp
def dfd (end, sen, start):
    data_disp = ddb.disp(end, sen, start)
    data_disp.drop(['depth', 'vel_xz', 'vel_xy'], axis=1, inplace=True)
    #data_disp['ts']=data_disp['ts'].dt.round('60min')
    data_disp ['ts'] = pd.to_datetime(data_disp.ts)
    data_disp.set_index('ts')
    return data_disp
###################################################################################################soms
def dfs (end, sen2, start):
    data_soms = ddb.somsdata(end, sen2, start)
    #data_soms['ts']=data_soms['ts'].dt.round('30min')
    data_soms ['ts'] = pd.to_datetime(data_soms.ts)
    data_soms.set_index('ts')
    
    return data_soms
###################################################################################################rainfall
def rainfal (start, end, gauge):
    import mysql.connector as sql

    db_connection = sql.connect(host='127.0.0.1', database='senslopedb', user='root', password='alienware091394')
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT * FROM senslopedb.{}".format(gauge))

    table_rows = db_cursor.fetchall()

    rainfall = pd.read_sql("SELECT * FROM senslopedb.{}".format(gauge), con=db_connection)
    rainfall = rainfall[(rainfall['timestamp'] >= start) & (rainfall['timestamp'] <= end)]
    rainfall.sort_values(['timestamp'],inplace = True)
    rainfall['timestamp']=rainfall['timestamp'].dt.round('30min')
    rainfall.drop(['name', 'batv1', 'batv2', 'cur', 'boostv1', 'boostv2', 'charge','csq', 'temp', 'hum', 'flashp'],axis=1, inplace =True)
    rainfall ['norm'] = (rainfall['r15m'] - rainfall['r15m'].min()) / (rainfall['r15m'].max() - rainfall['r15m'].min())

    return rainfall
#####################################################################################################groundata
def ground (start, end, groundmeas):
    import mysql.connector as sql

    db_connection = sql.connect(host='127.0.0.1', database='senslopedb', user='root', password='alienware091394')
    db_cursor = db_connection.cursor()
    db_cursor.execute("SELECT * FROM senslopedb.{}".format(groundmeas))

    table_rows = db_cursor.fetchall()

    ground = pd.read_sql("SELECT * FROM senslopedb.{}".format(groundmeas), con=db_connection)
    ground = ground[(ground['timestamp'] >= pd.to_datetime(start)) & (ground['timestamp'] <= pd.to_datetime(end))]
    ground.sort_values(['timestamp'],inplace = True)
    ground['timestamp']=ground['timestamp'].dt.round('30min')
    ground.drop(['meas_type', 'observer_name', 'weather', 'reliability'],axis=1, inplace =True)
    
    return ground
