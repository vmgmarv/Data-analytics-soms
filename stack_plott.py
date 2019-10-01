# -*- coding: utf-8 -*-
"""
Created on Thu Aug 02 14:09:27 2018
"""

import os
import sys
os.path.abspath('C:\Users\\meryl\\Documents\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\meryl\\Documents\updews-pycodes\Analysis')) #GroundDataAlertLib - line 21

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
plt.ion()

from datetime import timedelta
import numpy as np
import pandas as pd
#import seaborn

import ColumnPlotter as plotter
import genproc as proc
import querySenslopeDb as qdb
import rtwindow as rtw

mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

def nonrepeat_colors(ax,NUM_COLORS,color='jet'):
    cm = plt.get_cmap(color)
    ax.set_color_cycle([cm(1.*(NUM_COLORS-i-1)/NUM_COLORS) for i in range(NUM_COLORS)[::-1]])
    return ax

def zeroed(df, column):
    df['zeroed_'+column] = df[column] - df[column].values[0]
    return df

# surficial data
def get_surficial_df(site, start, end):

    query = "SELECT timestamp, site_id, crack_id, meas FROM gndmeas"
    query += " WHERE site_id = '%s'" % site
    query += " AND timestamp <= '%s'"% end
    query += " AND timestamp > '%s'" % start
    query += " ORDER BY timestamp"
    
    df = qdb.GetDBDataFrame(query)    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['crack_id'] = map(lambda x: x.upper(), df['crack_id'])
    
    marker_df = df.groupby('crack_id', as_index=False)
    df = marker_df.apply(zeroed, column='meas')
    
    return df

# surficial plot
def plot_surficial(ax, df, marker_lst):
    if marker_lst == 'all':
        marker_lst = set(df.crack_id)
    ax = nonrepeat_colors(ax,len(marker_lst))
    for marker in marker_lst:
        marker_df = df[df.crack_id == marker]
        ax.plot(marker_df.timestamp, marker_df.zeroed_meas, marker='o',
                label=marker, alpha=1)
    ax.set_ylabel('Displacement\n(cm)', fontsize='small')
    ax.set_title('Surficial Ground Displacement', fontsize='medium')
    ncol = (len(set(df.crack_id)) + 3) / 4
    ax.legend(loc='upper left', ncol=ncol, fontsize='x-small', fancybox = True, framealpha = 0.5)
    ax.grid()

# rainfall data
def get_rain_df(rain_gauge, start, end):
    rain_df = qdb.GetRawRainData(rain_gauge, fromTime=pd.to_datetime(start)-timedelta(3), toTime=end)
    
    rain_df = rain_df[rain_df.rain >= 0]
    rain_df = rain_df.set_index('ts')
    rain_df = rain_df.resample('30min').sum()
    
    rain_df['one'] = rain_df.rain.rolling(window=48, min_periods=1, center=False).sum()
    rain_df['one'] = np.round(rain_df.one, 2)
    rain_df['three'] = rain_df.rain.rolling(window=144, min_periods=1, center=False).sum()
    rain_df['three'] = np.round(rain_df.three, 2)
    
    rain_df = rain_df[(rain_df.index >= start) & (rain_df.index <= end)]
    rain_df = rain_df.reset_index()
    
    return rain_df

# rainfall plot
def plot_rain(ax, df, rain_gauge, plot_inst=True):
    ax.plot(df.ts, df.one, color='green', label='1-day cml', alpha=1)
    ax.plot(df.ts,df.three, color='blue', label='3-day cml', alpha=1)
    
    if max(list(df.one) + list(df.three)) >= 300:
        ax.set_ylim([0, 300])
    
    if plot_inst:
        ax2 = ax.twinx()
        width = float(0.004 * (max(df['ts']) - min(df['ts'])).days)
        ax2.bar(df['ts'].apply(lambda x: pltdates.date2num(x)), df.rain, width=width,alpha=0.1, color='k', label = '30min rainfall')
        ax2.xaxis_date()
    
    query = "SELECT * FROM rain_props where name = '%s'" %site
    twoyrmax = qdb.GetDBDataFrame(query)['max_rain_2year'].values[0]
    halfmax = twoyrmax/2
    
    ax.plot(df.ts, [halfmax]*len(df.ts), color='green', label='half of 2-yr max', alpha=1, linestyle='--')
    ax.plot(df.ts, [twoyrmax]*len(df.ts), color='blue', label='2-yr max', alpha=1, linestyle='--')
    
    ax.set_title("%s Rainfall Data" %rain_gauge.upper(), fontsize='medium')  
    ax.set_ylabel('1D, 3D Rain\n(mm)', fontsize='small')  
    ax.legend(loc='upper left', fontsize='x-small', fancybox = True, framealpha = 0.5)
    #ax.grid()
    
def plot_single_event(ax, ts, color='red'):
    ax.axvline(ts, color=color, linestyle='--', alpha=1)    
    
def plot_span(ax, start, end, color):
    ax.axvspan(start, end, facecolor=color, alpha=0.2, edgecolor=None,linewidth=0)
    
