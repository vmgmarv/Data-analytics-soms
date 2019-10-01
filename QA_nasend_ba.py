# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 10:10:47 2018
"""

import mysql.connector as sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


db_connection = sql.connect(host='192.168.150.75', database='senslopedb', user='pysys_local', password='NaCAhztBgYZ3HwTkvHwwGVtJn5sVMFgg')

read1 = db_connection.cursor()
read2 = db_connection.cursor()


tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

dyna_colors = [(22,82,109),(153,27,30),(248,153,29)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

for i in range(len(dyna_colors)):
    r_d,g_d,b_d = dyna_colors[i]
    dyna_colors[i] = (r_d / 255., g_d / 255., b_d / 255.)




def network_prefix(df):
    globe = ['05','06','15','16','17','25','26','27','35','36','37','45',
             '55','56','65','66','67','75','77','78','79','94','95','96','97']
    
    smart = ['00','07','08','09','10','11','12','13','14','18','19','20','21',
             '22','23','24','25','28','29','30','31','32','33','34','38','39',
             '40','41','42','43','44','46','47','48','49','50','51','70','81',
             '89','92','98','99']
    
    num = df.sim_num
    network_pref = num.apply(lambda x: x[-9] + x[-8])
    
    network_type = []
    for i in range (0, len(network_pref)):    
        if network_pref[i] in globe:
            network_type.append('GLOBE')
        elif network_pref[i] in smart:
            network_type.append('SMART')
        else:
            network_type.append('UNKNOWN')
        
    network_type = np.array(network_type)
    
    return network_type


def msg(start, end, num):
    read = db_connection.cursor()
    query = "SELECT * FROM comms_db.smsoutbox_users"
    query += " NATURAL JOIN comms_db.smsoutbox_user_status as us"
    query += " JOIN comms_db.user_mobile as um"
    query += " ON us.mobile_id = um.mobile_id"
    query += " WHERE ts_written BETWEEN '%s' AND '%s'" %(start, end)
    query += " AND sms_msg regexp 'alert level'"
    
    if num == 5:
        query += "AND send_status >= '5'"
    else:
        query += "AND send_status < '5'"
        
    
    read.execute(query)
    msg = pd.DataFrame(read.fetchall())
    msg.columns = read.column_names
    print num,query
    
    return msg

def count(df):
    col = ['outbox_id', 'sms_msg', 'web_status', 'priority']
    df.drop(col,axis=1, inplace =True)
    df = df.loc[:, ~df.columns.duplicated()]
    network = network_prefix(df)
    col = ['network']
    network = pd.DataFrame(network, columns = col)
    network_count = network['network'].value_counts()
    #network_count = pd.DataFrame(network_count, columns = col)
    ##############gsm
#    gsm_count = df['gsm_id'].value_counts().sort_index(ascending=True)
    #gsm_count = pd.DataFrame(gsm)
    
    return network_count#, gsm_count

def count_old(df,start,end):
    df = df[(df['timestamp_sent'] >= start) & (df['timestamp_sent'] <= end)]
    df = df[df['sms_msg'].str.contains("alert level") == True]
    #df.loc[df['gsm_id'] == 'UNKNOWN', ['send_status']] = 'FAIL'
    #####################################################################    
    unsent = df[(df['send_status'] == 'FAIL')]
    sent = df[(df['send_status'] == 'SENT')]
    network_sent = sent['gsm_id'].value_counts()
    network_unsent = unsent['gsm_id'].value_counts()  
                   
    return network_sent, network_unsent


#def ground_reminder(df,start,end):
#    df = df[(df['timestamp_sent'] >= start) & (df['timestamp_sent'] <= end)]
#    df = df[df['sms_msg'].str.contains("Inaasahan ang pagpapadala ng ground") == True]
#    pending = df[(df['send_stats'] != 'PENDING' )]
#    sent = df['send_status']
    
start = '2018-01-01'
end = '2018-12-23'
quarter = '3rd'
num = ''

#sent = msg(start, end, num = 5)
#unsent = msg(start, end, num = 0)
##
#network_sent = count(sent)
#network_unsent  = count(unsent)
#s3 = pd.Series([0], index=['UNKNWON'])
#network_unsent = network_unsent.append(s3)#deleteeee



df = pd.read_csv('smsoutbox.csv')
network_sent, network_unsent = count_old(df, start, end) 
network_sent = pd.DataFrame(network_sent)
network_sent = network_sent.sort_index()
network_sent.columns = ['gsm_id']
network_unsent = pd.DataFrame(network_unsent)
network_unsent = network_unsent.sort_index()
network_unsent.columns = ['gsm_id']

def plotting(network_sent, network_unsent, quarter):
    plt.rcParams['xtick.labelsize']=20
    plt.rcParams['ytick.labelsize']=20
    pos = list(range(len(network_unsent)))
    width = 0.25 
    fig, ax = plt.subplots(figsize=(10,8))

    plt.bar(pos, 
            network_unsent.gsm_id, 
            width, 
            alpha=0.5, 
            color=dyna_colors[1], 
            label='Unsent') 
    
    plt.bar([p + width for p in pos], 
            network_sent.gsm_id,
            width, 
            alpha=0.5, 
            color=dyna_colors[0], 
            label='Sent')
    
    for i in ax.patches:
        ax.text(i.get_x() + .05, i.get_height()-10, \
                str(round((i.get_height()/(network_sent.gsm_id.sum()+ network_unsent.gsm_id.sum()))* 100,2)) + '%', 
                   fontsize = 30, color = dyna_colors[2], fontweight='bold')
    
    # Set the y axis label
    ax.set_ylabel('Count', fontsize = 25)
    ax.set_xlabel('Network', fontsize = 25)
    # Set the chart's title
    ax.set_title('EWI GSM Sending Success Rate ({} quarter)'.format(quarter), fontsize = 30, y=1.02)
    
    # Set the position of the x ticks
    ax.set_xticks([p + 1 * width for p in pos])
    
    # Set the labels for the x ticks
    ax.set_xticklabels(network_unsent.index ,fontsize = 20)
    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*4)
    # Adding the legend and showing the plot
    plt.legend(fontsize = 25)
    #plt.legend(['Pre Score', 'Mid Score', 'Post Score'], loc='upper left')
    
    plt.show()

    return


#plotting(network_sent,network_unsent, quarter)






    
#################################################################################################################################
#plt.figure(1)
#plt.rcParams['xtick.labelsize']=20
#plt.rcParams['ytick.labelsize']=20
#pos = list(range(len(network_unsent['network']))) 
#width = 0.25 
#fig, ax = plt.subplots(figsize=(10,7))
#
#plt.bar(pos, 
#        #using df['pre_score'] data,
#        network_unsent['network'], 
#        # of width
#        width, 
#        # with alpha 0.5
#        alpha=0.5, 
#        # with color
#        color=dyna_colors[1], 
#        # with label the first value in first_name
#        label='Unsent') 
#
#plt.bar([p + width for p in pos], 
#        #using df['mid_score'] data,
#        network_sent['network'],
#        # of width
#        width, 
#        # with alpha 0.5
#        alpha=0.5, 
#        # with color
#        color=dyna_colors[0], 
#        # with label the second value in first_name
#        label='Sent')
#
#for i in ax.patches:
#    ax.text(i.get_x() + .10, i.get_height()-10, \
#            str(round((i.get_height()/(network_sent.network.sum()+ network_unsent.network.sum()))* 100,2)) + '%', fontsize = 20, color = dyna_colors[2])
#
## Set the y axis label
#ax.set_ylabel('Count', fontsize = 25)
#ax.set_xlabel('Network', fontsize = 25)
## Set the chart's title
#ax.set_title('Nagsend ba? (Last quarter)', fontsize = 30)
#
## Set the position of the x ticks
#ax.set_xticks([p + 1 * width for p in pos])
#
## Set the labels for the x ticks
#ax.set_xticklabels(network_unsent.index ,fontsize = 20)
## Setting the x-axis and y-axis limits
#plt.xlim(min(pos)-width, max(pos)+width*4)
## Adding the legend and showing the plot
#plt.legend(fontsize = 25)
##plt.legend(['Pre Score', 'Mid Score', 'Post Score'], loc='upper left')
#
#plt.show()
#
#################################################################################################################
#plt.figure(2)
#
#plt.rcParams['xtick.labelsize']=20
#plt.rcParams['ytick.labelsize']=20
#pos = list(range(len(gsm_sent['gsm_id']))) 
#width = 0.25 
#fig, ax = plt.subplots(figsize=(10,7))
#
#plt.bar(pos, 
#        #using df['pre_score'] data,
#        gsm_unsent['gsm_id'], 
#        # of width
#        width, 
#        # with alpha 0.5
#        alpha=0.5, 
#        # with color
#        color=dyna_colors[1], 
#        # with label the first value in first_name
#        label='Unsent') 
#
#plt.bar([p + width for p in pos], 
#        #using df['mid_score'] data,
#        gsm_sent['gsm_id'],
#        # of width
#        width, 
#        # with alpha 0.5
#        alpha=0.5, 
#        # with color
#        color=dyna_colors[0], 
#        # with label the second value in first_name
#        label='Sent')
#
#for i in ax.patches:
#    ax.text(i.get_x() + .01, i.get_height()-10, \
#            str(round((i.get_height()/(gsm_sent.gsm_id.sum() + gsm_unsent.gsm_id.sum()))* 100,2)) + '%', fontsize = 20, color = dyna_colors[2])
#
## Set the y axis label
#ax.set_ylabel('Count', fontsize = 25)
#ax.set_xlabel('gsm_id', fontsize = 25)
## Set the chart's title
#ax.set_title('Nagsend ba? (Last quarter)', fontsize = 30)
#
## Set the position of the x ticks
#ax.set_xticks([p + 0.17 * width for p in pos])
#
## Set the labels for the x ticks
#ax.set_xticklabels(gsm_unsent.index ,fontsize = 20)
## Setting the x-axis and y-axis limits
#plt.xlim(min(pos)-width, max(pos)+width*4)
## Adding the legend and showing the plot
#plt.legend(fontsize = 25)
##plt.legend(['Pre Score', 'Mid Score', 'Post Score'], loc='upper left')
#
#plt.show()
#
##plt.figure(1)
##
##ax = network['network'].value_counts().plot(kind = 'bar', color = dyna_colors[0], 
##            figsize = (10,7), fontsize = 25)
##ax.set_xticklabels(['Globe', 'Smart'], rotation = 0)
##ax.set_title('Sending Successful (Whole year)', fontsize = 30)
##ax.set_ylabel('Count', fontsize = 25)
##ax.set_xlabel('Network', fontsize = 25)
##
##
##for i in ax.patches:
##    ax.text(i.get_x() + .20, i.get_height()-15, \
##            str(round((i.get_height()/total_network)* 100,2)) + '%', fontsize = 20, color = dyna_colors[2])
##
##
##plt.figure(2)
##ax2 = df['gsm_id'].value_counts().sort_index(ascending=True).plot(kind = 'bar', color = dyna_colors[0], 
##            figsize = (10,7), fontsize = 25)
##ax2.set_xticklabels(['2', '3', '4', '5', '8', '9'], rotation = 0)
##ax2.set_title('Sending Successful (Whole year)', fontsize = 30)
##ax2.set_ylabel('Count', fontsize = 25)
##ax2.set_xlabel('Gsm_id', fontsize = 25)
##
##for i in ax2.patches:
##    ax2.text(i.get_x() + .15, i.get_height()-10, \
##            str(round((i.get_height()/total_gsm_id)* 100,2)) + '%', fontsize = 20, color = dyna_colors[2])
##################################################################################################################
