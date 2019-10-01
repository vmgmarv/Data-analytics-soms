##### IMPORTANT matplotlib declarations must always be FIRST to make sure that matplotlib works with cron-based automation
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
plt.ion()

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

import filepath

def col_pos(colpos_dfts):  
    colpos_dfts = colpos_dfts.drop_duplicates()
    cumsum_df = colpos_dfts[['mval1']].cumsum()
    colpos_dfts['cs_mval1'] = cumsum_df.mval1.values
               
    return np.round(colpos_dfts, 4)

#def compute_depth(colpos_dfts):
   # colpos_dfts = colpos_dfts.drop_duplicates()
   # cumsum_df = colpos_dfts[['depth']].cumsum()
   # cumsum_df['depth'] = cumsum_df['depth'] - min(cumsum_df.depth)
   # colpos_dfts['depth'] = cumsum_df['depth'].values
    #return np.round(colpos_dfts, 4)

def adjust_depth(colpos_dfts, max_depth):
    depth = max_depth - max(colpos_dfts['depth'].values)
    colpos_dfts['depth'] = colpos_dfts['depth'] + depth
    return colpos_dfts

def compute_colpos(window, config, monitoring_vel, fixpoint=''):
    if fixpoint == '':
        column_fix = config.io.column_fix
    else:
        column_fix = fixpoint
    
    colpos_df = monitoring_vel
    
   
    column_fix == 'bottom'
    
    if column_fix == 'top':
        colpos_df = colpos_df.sort('id', ascending = True)
    elif column_fix == 'bottom':
        colpos_df = colpos_df.sort('id', ascending = False)
    
    colpos_dfts = colpos_df.groupby('ts', as_index=False)
    colposdf = colpos_dfts.apply(col_pos)
    
    colposdf = colposdf.sort('id', ascending = True)
    #colpos_dfts = colposdf.groupby('ts', as_index=False)
    #colposdf = colpos_dfts.apply(compute_depth)
    
    #column_fix == 'bottom'
    #max_depth = max(colposdf['depth'].values)
   # colposdfts = colposdf.groupby('ts', as_index=False)
    #colposdf = colposdfts.apply(adjust_depth, max_depth=max_depth)
    
    #colposdf['depth'] = colposdf['depth'].apply(lambda x: -x)
    
    return colposdf

def nonrepeat_colors(ax,NUM_COLORS,color='gist_rainbow'):
    cm = plt.get_cmap(color)
    ax.set_color_cycle([cm(1.*(NUM_COLORS-i-1)/NUM_COLORS) for i in range(NUM_COLORS)[::-1]])
    return ax

def subplot_colpos(dfts, ax_xz, show_part_legend, config, colposTS):

    #current column position x
    curcolpos_x = dfts['depth'].values

    #current column position xz
    curax = ax_xz
    curcolpos_xz = dfts['cs_mval1'].apply(lambda x: x*1000).values
    curax.plot(curcolpos_xz,curcolpos_x,'.-')
    curax.set_xlabel('horizontal displacement, \n downslope(mm)')
    curax.set_ylabel('depth, m')
    

def plot_column_positions(df,end, show_part_legend, config, max_min_cml=''):
#==============================================================================
# 
#     DESCRIPTION
#     returns plot of xz and xy absolute displacements of each node
# 
#     INPUT
#     colname; array; list of sites
#     x; dataframe; vertical displacements
#     xz; dataframe; horizontal linear displacements along the planes defined by xa-za
#     xy; dataframe; horizontal linear displacements along the planes defined by xa-ya
#==============================================================================

    try:
        fig=plt.figure()
        ax_xz=fig.add_subplot(111)
    
        ax_xz=nonrepeat_colors(ax_xz,len(set(df.ts.values)),color='plasma')
    
        colposTS = pd.DataFrame({'ts': sorted(set(df.ts)), 'index': range(len(set(df.ts)))})
        
        dfts = df.groupby('ts', as_index=False)
        dfts.apply(subplot_colpos, ax_xz=ax_xz, show_part_legend=show_part_legend, config=config, colposTS=colposTS)
    
#        try:
#            max_min_cml = max_min_cml.apply(lambda x: x*1000)
#            xl = df.loc[(df.ts == end)&(df.id <= num_nodes)&(df.id >= 1)]['depth'].values[::-1]
#            ax_xz.fill_betweenx(xl, max_min_cml['xz_maxlist'].values, max_min_cml['xz_minlist'].values, where=max_min_cml['xz_maxlist'].values >= max_min_cml['xz_minlist'].values, facecolor='0.7',linewidth=0)
#            ax_xy.fill_betweenx(xl, max_min_cml['xy_maxlist'].values, max_min_cml['xy_minlist'].values, where=max_min_cml['xy_maxlist'].values >= max_min_cml['xy_minlist'].values, facecolor='0.7',linewidth=0)
#        except:
#            print 'error in plotting noise env'
    
        for tick in ax_xz.xaxis.get_minor_ticks():
            tick.label.set_rotation('vertical')
            tick.label.set_fontsize(10)
       
        for tick in ax_xz.xaxis.get_major_ticks():
            tick.label.set_rotation('vertical')
            tick.label.set_fontsize(10)
            
    
        plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.73)        
        #plt.suptitle(colname,fontsize='medium')
        ax_xz.grid(True)

    except:        
        print  "ERROR in plotting column position"
    return ax_xz
    
def main(dfs, window, config,show_part_legend = False, realtime=True, \
            plot_inc= False, end_mon=False):

    #colname = 'imuscm'
    #num_nodes = '15'
    #seg_len = '1'

    
    monitoring_vel =dfs.reset_index()[['ts', 'id', 'mval1']] ###dataframe
    #monitoring_vel = monitoring_vel.loc[(monitoring_vel.ts >= window.start)&(monitoring_vel.ts <= window.end)]#dataframe

    # compute column position
    colposdf = compute_colpos(window, config, monitoring_vel) ######youre here :P fix this

    # plot column position
    plot_column_positions(colposdf,window.end, show_part_legend, config)
    
    #lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    
    #plt.savefig(file_path['monitoring_output'] + colname + '_ColPos_' + \
    #        str(window.end.strftime('%Y-%m-%d_%H-%M')) + '.png', dpi=160, 
     #       facecolor='w', edgecolor='w', orientation='landscape', mode='w',
      #      bbox_extra_artists=(lgd,))

   # if file_path['event']:
    #    plt.savefig(file_path['event'] + colname + '_ColPos_' + \
     #           str(window.end.strftime('%Y-%m-%d_%H-%M')) + '.png', dpi=160,
      #          facecolor='w', edgecolor='w', orientation='landscape', mode='w',
       #         bbox_extra_artists=(lgd,))

####################################didnt include displacement plot offset
    

# 