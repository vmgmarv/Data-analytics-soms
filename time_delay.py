# -*- coding: utf-8 -*-
"""
Created on Thu Aug 02 11:14:03 2018
"""

import os
import sys
os.path.abspath('C:\Users\\meryl\\Desktop\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\meryl\\Documents\updews-pycodes-master\Analysis')) #GroundDataAlertLib - line 21
import pandas as pd
import regression as rg
import matplotlib.pyplot as plt
import ColumnPlotter as plter
import numpy as np
from datetime import datetime
import matplotlib.mlab as mlab
import scipy.stats as stats
import seaborn as sns
sns.set_style("darkgrid")

start = '2017-06-07'
end = '2017-07-07'
site = 'soms_hinsb'
gauge = 'rain_hinsb'
site_code = 'lay'

import mysql.connector as sql

db_connection = sql.connect(host='192.168.150.75', database='senslopedb', user='pysys_local', password='NaCAhztBgYZ3HwTkvHwwGVtJn5sVMFgg')
read1 = db_connection.cursor()

###########################################################################################################rainfall
def rainfall (start_time, end_time):
    #df = pd.read_sql("SELECT * FROM senslopedb.{}".format(gauge), con=db_connection)
    command1 = """
            SELECT * FROM senslopedb.rain_hinsb
        """
    read1.execute(command1) 

    df = pd.DataFrame(read1.fetchall())
    df.columns = read1.column_names
    df = df[(df['ts'] >= start) & (df['ts'] <= end)]
    df.sort_values(['ts'],inplace = True)
    df['ts']=df['ts'].dt.round('30min')
#    df.drop(['name', 'batv1', 'batv2', 'cur', 'boostv1', 'boostv2', 'charge','csq', 'temp', 'hum', 'flashp'],axis=1, inplace =True)
    df['norm'] = (df['rain'] - df['rain'].min()) / (df['rain'].max() - df['rain'].min())
    df ['roll_sum_1d'] = df['rain'].rolling(48).sum()
    df ['roll_sum_3d'] = df['rain'].rolling(144).sum()
    rainfall = df.reset_index(drop=True)
    
    return rainfall
##########################################################################################################rainfall_alert
def rain_alerts(site_code):
    df = pd.read_sql ("SELECT * FROM senslopedb.site_level_alert", con=db_connection)
    df = df[df.site == 'lay']
    df_alert = df[df.alert == 'r1']

    alert_dates = df_alert.timestamp
    alert_dates = pd.DataFrame(alert_dates)
    alert_dates = alert_dates.reset_index(drop=True)
    
    return alert_dates
###########################################################################################################soms
def soms_calibration (start_time, end_time, site_soms):
    command1 = """
            SELECT * FROM senslopedb.soms_hinsb
        """
    read1.execute(command1) 

    df = pd.DataFrame(read1.fetchall())
    df.columns = read1.column_names

    df = df[(df['ts'] >= start) & (df['ts'] <= end)]

    da = rg.regression_lay(df)

    da['ts']=da['ts'].dt.round('30min')
    x = da[da.type_num == 110]
    x.drop(['mval2', 'type_num'], axis=1, inplace=True)

    def roll_ave(dfg):
        dfg['vol_roll_mean'] = dfg['vol'].rolling(7).mean()
        return dfg
    soms = x.groupby('node_id').apply(roll_ave)
    ave = soms.vol.mean(0)
    return soms,ave

####################################################################################################processing
#rainfall = rainfall(start,end)
df,average = soms_calibration(start,end,site)

#sat = 0.31 * 0.75
#def ave(dfg):
#    dfg['ave'] = dfg['vol'].mean()
#    return dfg
#dfg = df.groupby('timestamp').apply(ave)
#
#final = pd.merge(rainfall, dfg, on = 'timestamp')
#
#data = pd.DataFrame({'rain':[]})
#data = pd.DataFrame({'rain':[]})
#data2 = pd.DataFrame({'soms':[]})
#store = []
#store2 = []
#
#for i, row in final.iterrows():
#    if row['roll_sum_1d'] >= 22:
#        d = row['timestamp']
#        store.append(d)
#
#data['rain'] = store
#for i, row in final.iterrows():
#    if row['ave'] >= sat:
#        d = row['timestamp']
#        store2.append(d)
#
#data2['soms'] = store2
#
#a = data.iloc[0]['rain']
#b = data2.iloc[0]['soms']
#c = b - a
#fig, ax1 = plt.subplots()
#ax1.plot(dfg.timestamp, dfg.ave, color = 'blue', linewidth = 2)
#ax1.set_xlabel('timestamp', fontsize = 30)
#ax1.tick_params(axis='both', which='major', labelsize=15)
## Make the y-axis label, ticks and tick labels match the line color.
#ax1.set_ylabel(r'$\theta[cm^3/cm^3]$',fontsize=30, color='b')
#ax1.tick_params('y', colors='b')
#plt.axhline(y = sat, linewidth = 4, color = 'b', linestyle = '--')
#
#ax2 = ax1.twinx()
#ax2.plot(rainfall.timestamp, rainfall.roll_sum_1d, color = 'red', linewidth = 2)
#ax2.set_ylabel(r'Rainfall$[mm]$', color='r', fontsize=30)
#ax2.tick_params('y', colors='r')
#ax2.tick_params(axis='both', which='major', labelsize=15)
#
#plt.axvspan(b, a, facecolor='g', alpha=0.5)
#
#plt.axhline(y = 22, linewidth = 4, color = 'r', linestyle = '--')
#fig.tight_layout()
##ax1.set_title('Time-delay = {}'.format(c), fontsize = 30)
#plt.show()

##########################################################################################################
rainfall_soms = pd.merge(rainfall, df, on = 'ts')
maximum = 0.33

final_sum1 = pd.DataFrame()
final_sum2 = pd.DataFrame()
n1 = 3
n2 = 4
n3 = 5
node  = rainfall_soms[rainfall_soms.node_id == n1]
df_final = node[node.vol >= maximum]

node2  = rainfall_soms[rainfall_soms.node_id == n2]
df_final2 = node2[node2.vol >= maximum]

node3  = rainfall_soms[rainfall_soms.node_id == n3]
df_final3 = node3[node3.vol >= maximum]
for n in range (3,10):
    node  = rainfall_soms[rainfall_soms.node_id == n]
    df_final = node[node.vol >= maximum]
    
    summary = df_final.describe([.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.99])
    summ = summary.drop(['rain', 'r24h', 'norm', 'id', 'mval1', 'grav', 'vol', 'vol_roll_mean'], axis = 1) ##########drops specific columns
    summ_final = summ.drop(['count', 'std']) #####################drops specific rows
    
    d1 = {'{}'.format(n) : summ_final.roll_sum_1d} ###########1 Day
    d2 = {'{}'.format(n) : summ_final.roll_sum_3d} ###########3 Day
    dfm1 = pd.DataFrame(data=d1)
    dfm2 = pd.DataFrame(data=d2)

    final_sum1 = pd.concat([final_sum1, dfm1], axis = 1) #########1 Day
    final_sum2 = pd.concat([final_sum2, dfm2], axis = 1) #########3 Day

#writer = pd.ExcelWriter('{}_rain_soms.xlsx'.format(site))
#df_final.to_excel(writer,'1 Day')
##final_sum2.to_excel(writer,'3 Days')
#writer.save()

n_bins = 30

x = df_final.roll_sum_1d.dropna().sort_values().tolist()
xmean = np.mean(x)
xstd = np.std(x)
fit_x = stats.norm.pdf(x, xmean, xstd)

y = df_final2.roll_sum_1d.dropna().sort_values().tolist()
ymean = np.mean(y)
ystd = np.std(y)
fit_y = stats.norm.pdf(y, ymean, ystd)

z = df_final3.roll_sum_1d.dropna().sort_values().tolist()
zmean = np.mean(z)
zstd = np.std(z)
fit_z = stats.norm.pdf(z, zmean, zstd)


fig2, axs = plt.subplots(1, 3, sharey = False, tight_layout = False)
plt.suptitle('Histogram 1-Day Cumulative Rainfall', fontsize = 30)

axs[0].hist(x, bins=n_bins, normed = 1, color = 'green', rwidth=0.8, alpha = 1)
axs[0].plot(x, fit_x, 'b--', label = 'normal dist')
axs[0].tick_params(axis='both', which='major', labelsize=15)
axs[0].set_title('Node {}'.format(n1), fontsize = 20)
axs[0].axvline(x=65.5725, linewidth=2, color = 'r', label = 'Threshold (65.5725 mm)')

axs[1].hist(y, bins=n_bins, normed = 1, color = 'green', rwidth=0.8, alpha = 1)
axs[1].plot(y, fit_y, 'b--', label = 'normal dist')
axs[1].tick_params(axis='both', which='major', labelsize=15)
axs[1].set_title('Node {}'.format(n2), fontsize = 20)
axs[1].axvline(x=65.5725, linewidth=2, color = 'r', label = 'Threshold (65.5725 mm)')

axs[2].hist(z, bins=n_bins, normed = 1, color = 'green', rwidth=0.8, alpha = 1)
axs[2].plot(z, fit_z, 'b--', label = 'normal dist')
axs[2].tick_params(axis='both', which='major', labelsize=15)
axs[2].set_title('Node {}'.format(n3), fontsize = 20)
axs[2].axvline(x=65.5725, linewidth=2, color = 'r', label = 'Threshold (65.5725 mm)')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
fig2.text(0.5, 0.01, r'1-Day Rainfall [$mm$]', fontsize = 30, ha='center')
fig2.text(0.04, 0.5, 'Probability',fontsize= 30, va='center', rotation='vertical')

leg2 = plt.legend(loc='best', bbox_to_anchor=(0.4, 1.01), ncol=1, framealpha = 1, frameon = True, borderpad = 1,
          shadow=True, fancybox=True, fontsize=18)
