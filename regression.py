#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:17:44 2018
"""

import os
import sys
os.path.abspath('C:\Users\\meryl\\Documents\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\meryl\\Documents\updews-pycodes\Analysis')) #GroundDataAlertLib - line 21
import querySenslopeDb as q
import matplotlib.pyplot as plt
import pandas as pd
import data_mine_disp as dp
import numpy as np
import numpy.polynomial.polynomial as poly
import s_d_db as sd
from matplotlib import colors as mcolors
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

##################################################################################################################
def regression (df):
    #silt = [0, 0.08, 0.103, 0.157, 0.193, 0.21]
    #resp_silt = [187.5, 351.5, 416, 572, 702, 809.5]

    silt = [0.02, 0.08, 0.103, 0.157, 0.193, 0.21, 1]
    resp_silt = [847, 911, 949.5, 1043, 1120.5, 1185.5, 1216.5]
    
    d = {'grav' : silt, 'mval1':resp_silt}
    dg = pd.DataFrame(d)

    y = dg['grav']
    x = dg['mval1']
    m,b = np.polyfit(x, y, 1)
    
    fit = np.polyfit(x,y,3)
    fit_fn = np.poly1d(fit) 
#    
#    plt.scatter(x,y, label = 'grav')
#    plt.title('Regression(LABORATORY)',fontsize=30)
#    plt.xlabel('mval1 (raw)',fontsize=20)
#    plt.ylabel('grav (%gram)',fontsize=20)
#    plt.plot(x, fit_fn(x), '--k', label = 'regression')
#    plt.legend()
#    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
#    df.loc[(df['mval1'] > 1200), 'mval1'] = 1200
    
    df['grav'] = ((df['mval1'] * m) + b)
    df['vol'] = (df['grav'] * 1.3)
    xc = (0.098 - b) / m
    return df
#################################################################################0###############################
def regression2 (data):
    '''
    Calibration for Brgy Laygayon
    '''
    
    data.loc[(data['mval1'] > 1185), 'mval1'] = 1185
    data.loc[(data['mval1'] < 750), 'mval1'] = 750
########################################################################laysa(clay)
    d0 = pd.DataFrame(np.random.randint(1,98,size=(100)), columns = ['A'])
    d0['A'] = d0['A'] / 1000
    d0['B'] = pd.DataFrame(np.random.randint(699,980,size=(1000)))

    d1 = pd.DataFrame(np.random.randint(98,160,size=(100)), columns = ['A'])
    d1['A'] = d1['A'] / 1000
    d1['B'] = pd.DataFrame(np.random.randint(980,1030,size=(1000)))

    d2 = pd.DataFrame(np.random.randint(161,240,size=(200)), columns = ['A'])
    d2['A'] = d2['A'] / 1000
    d2['B'] = pd.DataFrame(np.random.randint(1031,1062,size=(1000)))

    d3 = pd.DataFrame(np.random.randint(241,330,size=(100)), columns = ['A'])
    d3['A'] = d3['A'] / 1000
    d3['B'] = pd.DataFrame(np.random.randint(1063,1142,size=(100)))

    d4 = pd.DataFrame(np.random.randint(331,480,size=(100)), columns = ['A'])
    d4['A'] = d4['A'] / 1000
    d4['B'] = pd.DataFrame(np.random.randint(1143,1184,size=(100)))

    d5 = pd.DataFrame(np.random.randint(481,520,size=(100)), columns = ['A'])
    d5['A'] = d5['A'] / 1000
    d5['B'] = pd.DataFrame(np.random.randint(1182,1184,size=(100)))

    d6 = pd.DataFrame(np.random.randint(521,580,size=(100)), columns = ['A'])
    d6['A'] = d6['A'] / 1000
    d6['B'] = pd.DataFrame(np.random.randint(1185,1194,size=(100)))

    d7 = pd.DataFrame(np.random.randint(581,1000,size=(100)), columns = ['A'])
    d7['A'] = d7['A'] / 1000
    d7['B'] = pd.DataFrame(np.random.randint(1196,1217,size=(100)))

    frames = [d0, d1, d2, d3, d4, d5, d6, d7]

    result = pd.concat(frames)

    data_x = result['B']
    data_y = result['A']

    x = np.array(data_x)
    y = np.array(data_y)

    z = np.polyfit(x, y, 3)

    p = np.poly1d(z)
    
    data['grav'] = p(data['mval1'])
    data['vol'] = data['grav'] / 1.21 
    #data.loc[(data['id'] >= 4), 'grav'] = 0.470989
    data.loc[(data['vol'] >= 0.465651), 'vol'] = 0.465651
    
    return data

def regression3 (data):
    '''
    Calibration for Brgy. Hinabangan
    '''
    data.loc[(data['mval1'] > 1181), 'mval1'] = 1181
    data.loc[(data['mval1'] < 896), 'mval1'] = 897


########################################################################hinsb

    d0 = pd.DataFrame(np.random.randint(1,123,size=(100)), columns = ['A'])
    d0['A'] = d0['A'] / 1000  ####grav
    d0['B'] = pd.DataFrame(np.random.randint(717,897,size=(1000))) ###sensor

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
    data_y = result['A']
    
    x = np.array(data_x)
    y = np.array(data_y)
    
    z = np.polyfit(x, y, 3)

    p = np.poly1d(z)
    
    data['grav'] = p(data['mval1'])
    data['vol'] = data['grav'] / 1.48
    #data.loc[(data['id'] >= 5), 'grav'] = 0.501362
    data.loc[(data['vol'] >= 0.345373), 'vol'] = 0.345373
    
    return data
#
def stackplot_soms_ground(groundmeas, start, end):
    
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
    ground = ground[ground.site_id == 'LAY']
    cracka = ground [ground.crack_id == 'A']
    cracka ['norm'] = (cracka['meas'] - cracka['meas'].min()) / (cracka['meas'].max() - cracka['meas'].min()) - 0.1
    crackb = ground [ground.crack_id == 'B']
    crackb ['norm'] = (crackb['meas'] - crackb['meas'].min()) / (crackb['meas'].max() - crackb['meas'].min()) - 0.1
    crackc = ground [ground.crack_id == 'C']
    crackc ['norm'] = (crackc['meas'] - crackc['meas'].min()) / (crackc['meas'].max() - crackc['meas'].min()) - 0.1
    crackd = ground [ground.crack_id == 'D']
    crackd ['norm'] = (crackd['meas'] - crackd['meas'].min()) / (crackd['meas'].max() - crackd['meas'].min()) - 0.1
    cracke = ground [ground.crack_id == 'E']
    cracke ['norm'] = (cracke['meas'] - cracke['meas'].min()) / (cracke['meas'].max() - cracke['meas'].min()) - 0.1

    day_end = datetime.strptime(end, '%Y-%m-%d')
    date_end = day_end - timedelta(days=30)
    date_end.strftime('%Y-%m-%d')

    day_start = datetime.strptime(start, '%Y-%m-%d')
    date_start = day_start - timedelta(days=35)
    date_start.strftime('%Y-%m-%d')
    fig2, (axes2) = plt.subplots(5, 1, sharex = True, sharey = True)
    fig2.subplots_adjust(hspace=0)
    t =  (5/2) + 0.5
    plt.text(date_start, t, '0 to 1', fontsize = 14, rotation = 90)
    axes2[0].set_title('Ground Meas (NORMALIZED)\nSite {}\n {} to {}'.format(site, start, end), fontsize = 14)
    axes2[0].plot(cracka.timestamp, cracka.norm, color = 'orange', label = 'Crack A')
    axes2[0].text(end, 0.16, 'Crack A', style='oblique',
         bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5})
    axes2[1].plot(crackb.timestamp, crackb.norm, color = 'orange', label = 'Crack B')
    axes2[1].text(end, 0.16, 'Crack B', style='oblique',
         bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5})
    axes2[2].plot(crackc.timestamp, crackc.norm, color = 'orange', label = 'Crack C')
    axes2[2].text(end, 0.16, 'Crack C', style='oblique',
         bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5})
    axes2[3].plot(crackd.timestamp, crackd.norm, color = 'orange', label = 'Crack D')
    axes2[3].text(end, 0.16, 'Crack D', style='oblique',
         bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5})
    axes2[4].plot(cracke.timestamp, cracke.norm, color = 'orange', label = 'Crack E')
    axes2[4].text(end, 0.16, 'Crack E', style='oblique',
         bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5})
    
    return
