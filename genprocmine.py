# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:40:57 2017
"""
import os
import sys
os.path.abspath('C:\Users\\Vidal Marvin Gabriel\\Desktop\DYNASLOPE\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\Vidal Marvin Gabriel\\Desktop\DYNASLOPE\updews-pycodes\Analysis')) #GroundDataAlertLib - line 21
import pandas as pd
import numpy as np
from datetime import timedelta
from pandas.stats.api import ols

import querySenslopeDb as q
import filterSensorData as flt
import errorAnalysis as err

class procdata:
    def __init__ (self, colprops, disp_vel, max_min_df, max_min_cml):
        self.colprops = colprops
        self.disp_vel = disp_vel
        self.max_min_df = max_min_df
        self.max_min_cml = max_min_cml
        

def GetNodesWithNoInitialData(df,num_nodes,offsetstart):
    allnodes=np.arange(1,num_nodes+1)*1.
    with_init_val=df[df.ts<offsetstart+timedelta(hours=0.5)]['id'].values
    no_init_val=allnodes[np.in1d(allnodes, with_init_val, invert=True)]
    return no_init_val

def GetNodesWithNoData(df, num_nodes):
    allnodes = np.arange(1,num_nodes+1)
    withval = sorted(set(df.id))
    noval = allnodes[np.in1d(allnodes, withval, invert=True)]
    return noval

def resamplenode(df, window):
    blank_df = pd.DataFrame({'ts': [window.end,window.offsetstart], 'id': [df['id'].values[0]]*2, 'name': [df['name'].values[0]]*2}).set_index('ts')
    df = df.append(blank_df)
    df = df.reset_index().drop_duplicates(['ts','id']).set_index('ts')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending = True)
    df = df.resample('30Min', base=0, how='pad')
    df = df.reset_index(level=1)
    return df   

def fill_smooth(df, offsetstart, end, roll_window_numpts, to_smooth, to_fill):    
    if to_fill:
        # filling NAN values
        df = df.fillna(method = 'pad')
        
        #Checking, resolving and reporting fill process    
#        print 'Post-filling report: '
        if df.isnull().values.any():
            for n in ['id', 'mval1']:
                if df[n].isnull().values.all():
#                    print '     ',n, 'NaN all values'
                    df[n]=0
                elif np.isnan(df[n].values[0]):
#                    print '     ',n, 'NaN 1st value'
                    df[n]=df[n].fillna(method='bfill')
#        else: 
#            print '     All numerical values.'

    #dropping rows outside monitoring window
    df=df[(df.index>=offsetstart)&(df.index<=end)]
    
    if to_smooth and len(df)>1:
        df=pd.rolling_mean(df,window=roll_window_numpts,min_periods=1)[roll_window_numpts-1:]
        return df
    else:
        return df 

def genproc(col, window, config, fixpoint, realtime=False, comp_vel=True):
    
    monitoring = q.GetSomsData(col, window.offsetstart, window.end)
    
 
    
    try:
        LastGoodData = q.GetLastGoodData(monitoring,col.nos)
        q.PushLastGoodData(LastGoodData,col.name)		
        LastGoodData = q.GetLastGoodDataFromDb(col.name)
    except:	
        LastGoodData = q.GetLastGoodDataFromDb(col.name)
   
    #identify the node ids with no data at start of monitoring window
    NodesNoInitVal=GetNodesWithNoInitialData(monitoring,col.nos,window.offsetstart)
    
    #get last good data prior to the monitoring window (LGDPM)
    if len(NodesNoInitVal) != 0:
        lgdpm = q.GetSingleLGDPM(col.name, NodesNoInitVal, window.offsetstart)
        if len(lgdpm) != 0:
            lgdpm = flt.applyFilters(lgdpm)
            lgdpm = lgdpm.sort_index(ascending = False).drop_duplicates('id')
        
        if len(lgdpm) != 0:
            monitoring=monitoring.append(lgdpm)
        
    monitoring = monitoring.loc[monitoring.id <= col.nos]
    
    invalid_nodes = q.GetNodeStatus(1)
    invalid_nodes = invalid_nodes[invalid_nodes.site == col.name]['node'].values
    monitoring = monitoring.loc[~monitoring.id.isin(invalid_nodes)]
    
    #assigns timestamps from LGD to be timestamp of offsetstart
    monitoring.loc[(monitoring.ts < window.offsetstart)|(pd.isnull(monitoring.ts)), ['ts']] = window.offsetstart

    nodes_noval = GetNodesWithNoData(monitoring, col.nos)
    nodes_nodata = pd.DataFrame({'name': [0]*len(nodes_noval), 'id': nodes_noval,
                'ts': [window.offsetstart]*len(nodes_noval)})
    monitoring = monitoring.append(nodes_nodata)

    max_min_df, max_min_cml = err.cml_noise_profiling(monitoring, config, fixpoint, col.nos)

    monitoring = monitoring.drop_duplicates(['ts', 'id'])
    monitoring = monitoring.set_index('ts')
        
    #resamples xz and xy values per node using forward fill
    monitoring = monitoring.groupby('id').apply(resamplenode, window = window).reset_index(level=1).set_index('ts')
    
    nodal_proc_monitoring = monitoring.groupby('id')
    
    if not realtime:
        to_smooth = config.io.to_smooth
        to_fill = config.io.to_fill
    else:
        to_smooth = config.io.rt_to_smooth
        to_fill = config.io.rt_to_fill
    
    filled_smoothened = nodal_proc_monitoring.apply(fill_smooth, offsetstart=window.offsetstart, end=window.end, roll_window_numpts=window.numpts, to_smooth=to_smooth, to_fill=to_fill)
    filled_smoothened = filled_smoothened[['xz', 'xy', 'x', 'y', 'z', 'name']].reset_index()
            
    filled_smoothened['depth'] = filled_smoothened['x']/np.abs(filled_smoothened['x']) * np.sqrt(col.seglen**2 - filled_smoothened['xz']**2 - filled_smoothened['xy']**2)
    filled_smoothened['depth'] = filled_smoothened['depth'].fillna(value=col.seglen)

    monitoring = filled_smoothened.set_index('ts')   

    
    return procdata(col,disp_vel.sort(),max_min_df,max_min_cml)