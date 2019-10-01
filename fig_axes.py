# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 17:05:16 2018
"""

import os
import sys
os.path.abspath('C:\Users\meryl\Documents\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\meryl\Documents\updews-pycodes')) #GroundDataAlertLib - l
import querySenslopeDb as q
import matplotlib.pyplot as plt
import pandas as pd
import data_mine_disp as dp
import numpy as np
import ColumnPlotter as plter
import s_d_db as sd
from matplotlib import colors as mcolors
from datetime import datetime, timedelta

start = '2016-06-08'
end = '2016-09-26'
site = 'imuscm'
sen = 'imusc'

###################################################################################rainfall dataframe
import mysql.connector as sql


db_connection = sql.connect(host='192.168.150.129', database='senslopedb', user='root', password='senslope')
db_cursor = db_connection.cursor()
db_cursor.execute("SELECT * FROM senslopedb.partaw")

table_rows = db_cursor.fetchall()

rainfall = pd.read_sql("SELECT * FROM senslopedb.partaw", con=db_connection)
rainfall = rainfall[(rainfall['timestamp'] >= start) & (rainfall['timestamp'] <= end)]
rainfall.sort_values(['timestamp'],inplace = True)
rainfall['timestamp']=rainfall['timestamp'].dt.round('30min')
rainfall.drop(['name', 'batv1', 'batv2', 'cur', 'boostv1', 'boostv2', 'charge','csq', 'temp', 'hum', 'flashp'],axis=1, inplace =True)
rainfall ['normr24h'] = (rainfall['r24h'] - rainfall['r24h'].min()) / (rainfall['r24h'].max() - rainfall['r24h'].min())
rainfall ['normr15m'] = (rainfall['r15m'] - rainfall['r15m'].min()) / (rainfall['r15m'].max() - rainfall['r15m'].min())

#######################################################################################soms
 
df = sd.dfs(end, site, start)
df.sort_values(['ts'],inplace = True)
df['ts']=df['ts'].dt.round('30min')

dfg = df.groupby(['ts','id']).mean()['mval1']
dfg = pd.DataFrame(dfg)
my_columns = ["mval1"]
dfg.columns = my_columns
dfg.reset_index(inplace = True)
finalsoms = pd.DataFrame()
normalize = pd.DataFrame()
finaldisp = pd.DataFrame()



for n in range (1,16):
    dfid = dfg[dfg.id == n]
    id = dfid['id']
    ts = dfid['ts']
    mval1 = dfid['mval1']
    if dfid.empty:
       dfid['mval1_dif'] = 0
       dfid = dfid['mval1_dif']
                
       d = {'ts' : ts, 'id': id, 'mval1':mval1, 'mval1_dif': dfid}
       dfm = pd.DataFrame(data=d)
    
    else:
       dfid['mval1_dif'] = (dfid['mval1'] - dfid['mval1'].values[0])
       #dfid = dfid['mval1_dif']
       dfid ['norm'] = (dfid['mval1_dif'] - dfid['mval1_dif'].min()) / (dfid['mval1_dif'].max() - dfid['mval1_dif'].min())
       #norm = dfid['norm']     
       d = {'ts' : dfid.ts, 'id': dfid.id, 'mval1':dfid.mval1, 'mval1_dif': dfid.mval1_dif, 'norm':dfid.norm}
       dfm = pd.DataFrame(data=d)  
       
       finalsoms = pd.concat([finalsoms, dfm]).drop_duplicates()
       
finalsoms.set_index('ts', inplace =True)

#######################################################################################displacement
disp = sd.dfd(end, sen, start)

for n in range (1,16):
    dfid = disp[disp.id == n]
    id = dfid['id']
    ts = dfid['ts']
    xz = dfid['xz']
    xy = dfid['xy']
    if dfid.empty:
       dfid['xz_dif'] = 0
       dfid['xy_dif'] = 0
       xz = dfid['xz_dif']
       xy = dfid['xy_dif']
                
       d = {'ts' : ts, 'id': id, 'xz_dif':xz, 'xy_dif': xy}
       dfm = pd.DataFrame(data=d)
    
    else:
       dfid['xz_dif'] = (dfid['xz'] - dfid['xz'].values[0])
       dfid['xy_dif'] = (dfid['xy'] - dfid['xy'].values[0])
       #dfid = dfid['mval1_dif']
       dfid ['norm_xz'] = (dfid['xz_dif'] - dfid['xz_dif'].min()) / (dfid['xz_dif'].max() - dfid['xz_dif'].min())
       dfid ['norm_xy'] = (dfid['xy_dif'] - dfid['xy_dif'].min()) / (dfid['xy_dif'].max() - dfid['xy_dif'].min())
       #norm = dfid['norm']     
       d = {'ts' : dfid.ts, 'id': dfid.id, 'xz':dfid.xz,'xy':dfid.xy, 'norm_xz':dfid.norm_xz, 'norm_xy':dfid.norm_xy}
       dfm = pd.DataFrame(data=d)    
       
       finaldisp = pd.concat([finaldisp, dfm]).drop_duplicates()
       
finaldisp.set_index('ts', inplace =True)
final = pd.merge(left=finalsoms, right=finaldisp, left_index=True, right_index=True, how='left')
#########################################################################################       
# =============================================================================
fig1, (axs) = plt.subplots(nrows = 15, sharex = True)
fig1.subplots_adjust(hspace = 0, wspace=.001)
fig1.suptitle('Site {} (xz)'.format(sen))

# ============================================================================= ###################rainfall################################
# fig1.text(0.1, 0.5, 'Normalized value (0-1)', va='center', rotation='vertical')
# fig1.text(0.5, 0.04, 'Timestamp', ha='center')
# 
# axs[0].plot(rainfall.timestamp,rainfall.normr24h,color = 'b', label = 'rain(r24h)')
# axs[0].yaxis.set_visible(False)
# #axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
# axs[0].text(0.98,0.75,'Rain(r24)',
#         horizontalalignment='center',bbox={'facecolor':'red', 'alpha':0.1, 'pad':4},
#         transform=axs[0].transAxes)
# #axs2 = axs.twinx() ##############twin ax
# axs[1].plot(rainfall.timestamp,rainfall.normr15m,color = 'r', label = 'rain(r15m)')
# axs[1].yaxis.set_visible(False)
# #axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
# axs[1].text(0.98,0.75,'Rain(r15m)',
#         horizontalalignment='center',bbox={'facecolor':'red', 'alpha':0.1, 'pad':3},
#         transform=axs[1].transAxes)
# =============================================================================#############################################################
############################################################xz vs moist
a_soms = pd.to_datetime('2016-07-23')
b_soms = pd.to_datetime('2016-08-06')

a_disp = pd.to_datetime ('2016-08-15')
b_disp = pd.to_datetime('2016-08-21')


for i in range (1,16):
    n1 = final[final.id_x== i]
    n2 = final[final.id_y== i]
    axs[i-1].plot(n1.index,n1['norm'], color = 'orange', label = "moist".format(i))
    axs[i-1].plot(n2.index,n2['norm_xz'], color = 'green', label = "disp".format(i))
    axs[i-1].yaxis.set_visible(False)
    axs[i-1].spines['right'].set_color('none')
    axs[i-1].spines['left'].set_color('none')
    
    axs[i-1].text(0.96, 0.68, 'node{}'.format(i),
        fontsize=12, transform=axs[i-1].transAxes, bbox = {'facecolor':'blue', 'alpha':0.1, 'pad':4})
for ax in axs [0:]:
    ax.axvspan(a_disp, b_disp, color='r', alpha=0.2, lw=0)
for ax in axs[2:]:
    ax.axvspan(a_soms, b_soms, color='violet', alpha=0.2, lw=0)

legend = plt.legend(fontsize = 15, bbox_to_anchor= (-0.01,8.5), loc="center right", borderaxespad=0, fancybox = True)
legend.get_frame().set_facecolor('gray')
legend.get_frame().set_alpha(0.1)
##########################################################xy vs moist
