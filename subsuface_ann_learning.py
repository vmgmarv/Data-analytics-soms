# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 08:54:15 2018
"""

import SlopeInclinometers as si
import os
import sys
os.path.abspath('C:\Users\\meryl\\Desktop\mycodes') #GroundDataAlertLib - line 19
sys.path.insert(1,os.path.abspath('C:\Users\\meryl\\Documents\updews-pycodes-master\Analysis')) #GroundDataAlertLib - line 21
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from scipy import stats
from sklearn.neural_network import MLPRegressor


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

def GetSubsurfaceData(time_start,time_end,sensor_column,node_id,compute_vel = False):
    '''
    Gets the displacement data frame given the sensor name, node and timestamps
    
    Parameters
    ------------------------------
    time_start - string
        Desired start time
    time_end - string
        Desired end time
    sensor_column - string
        Name of sensor column
    node_id - int
        Node number of interest
    compute_vel - Boleean
        Adds velocity if True
    
    Returns
    -------------------------------
    subsurface_data - pd.DataFrame()
        Subsurface data frame
    sensor_column - string
        Name of sensor column
    '''
    
    #### Convert into tuple timestamps the given times
    timestamp = (pd.to_datetime(time_start),pd.to_datetime(time_end))
    
    #### Call the GetDispDataFrame from slope inclinometers
    subsurface_data,name = si.GetDispDataFrame(timestamp,sensor_column,compute_vel)
    
    #### Embed name to data frame
    subsurface_data['name'] = sensor_column
    
    return subsurface_data[subsurface_data.id == node_id][['ts','id','name','xz']]

def GetFeatureVectors2(subsurface_data,m,train_ratio,ave_window = '1H',smooth_window = 3):
    '''
    Creates the normalized input feature vectors and output vector from the time series divided according to train and test set
    
    Parameters
    -----------------------------
    subsurface_data - pd.DataFrame()
        DataFrame with displacement and time column
    m - int
        Embedding dimension
    train_ratio - float
        Desire ratio between training and testing sets
    ave_window - int
        Moving average window
    
    Returns
    -----------------------------
    train_x, train_y, test_x, test_y - np.array
    '''
    
    #### Get smoothened and resampled displacement data
    subsurface_data_xz = subsurface_data[['ts','xz']].resample(ave_window,label = 'right',closed = 'right',on = 'ts').mean().rolling(smooth_window).mean()

    #### Get displacement column
    subsurface_data_xz['disp'] = subsurface_data_xz.xz - subsurface_data_xz.xz.shift(1)
    displacement = subsurface_data_xz.disp.dropna().values

    #### Normalize displacement values
    displacement = (displacement - min(displacement))/(max(displacement) - min(displacement))
    
    #### Initialize results list
    X = []
    Y = []
    
    #### Iterate through the data frame based on m
    print 'len(displacement)', len(displacement)
    for i in range(len(displacement) - m):
        
        #### Add to list
        X.append(displacement[i:i+m])
       
        Y.append(displacement[i+m])
    
    
    #### Convert results to array
    X = np.array(X)
    Y = np.array(Y)
    
    #### Get last index of training set
    last_index = int(len(X)*train_ratio)
    print last_index
    
    #### Split into training set and test set
    train_x = X[0:last_index]
    train_y = Y[0:last_index]
    test_x = X[last_index:]
    test_y = Y[last_index:]
    
    return train_x, train_y, test_x, test_y, X, Y


def GetOptimalCLF2(train_x,train_y,rand_starts = 8):
    '''
    Gets the optimal CLF function based on fixed settings
    
    Parameters
    ------------------------
    train_x - np.array
        Training feature vectors
    train_y - np.array
        Training label vectors
    rand_starts - int
        Number of random starts to do
        Default - 8 for 95% confidence and best 30%
    
    Returns
    ------------------------
    max_clf - sklearn function
        Optimal trained artificial neuron network
    '''
    
    #### Get number of feature inputs of training vector
    n_input = train_x.shape[1]
    
    #### Set initial loss value
    min_loss = 1e10
    
    #### Perform number of trainings according to random start set
    for i in range(rand_starts):
        
        #### Print current status
        print "Iteration number {}".format(i+1)
        
        #### Initialize ANN network
        clf = MLPRegressor(hidden_layer_sizes = (10,10), activation = 'relu',solver = 'sgd', 
                           learning_rate = 'adaptive', max_iter = 100000000,tol = 1e-10,
                           early_stopping = True, validation_fraction = 1/3.)
    
        #### Fit data
        clf.fit(train_x,train_y)
        
        #### Get current loss
        cur_loss = clf.loss_
        print 'min_loss = ', min_loss
        print 'cur_loss = ', cur_loss
        #### Save current clf if loss is minimum
        if cur_loss < min_loss:
            
            #### Set min_loss to a new value
            min_loss = cur_loss
            
            #### Set max_clf to new value
            max_clf = clf
    print 'lowest loss =', cur_loss
    return max_clf

def GetTrainTestPlotResult2(subsurface_data,clf,train_x,train_y,test_x,test_y,ave_window = '4H',smooth_window = 12):
    '''
    Gets the superimposed plots of the actual and predicted training and test data via the trained ANN.
    Computes the rmse error of the classifier.
    
    Parameters
    -------------------
    subsurface_data - pd.DataFrame()
        Subsurface data frame
    clf - sklearn classifier
        Trained ANN
    train_x - np.array
        Training feature vectors
    train_y - np.array
        Training label vectors
    test_x - np.array
        Test feature vectors
    test_y - np.array
        Test label vectors
    ave_window - int
        Moving average window
    
    Returns
    -------------------
    test_rmse - root mean square error of the classifier
    '''
    
    #### Get data predictions of ANN
    pred_train = clf.predict(train_x)
    pred_test = clf.predict(test_x)
    
    #### Get starting indexes for plotting
    start_index = len(subsurface_data) - (len(train_y) + len(test_y))
    
    ############################################
    #### Plot displacement data predictions ####
    ############################################
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot measured data points
    ax.plot(subsurface_data.ts.values[start_index:],np.concatenate((train_y,test_y)),label = 'Measured',lw = 3.0)
    
    #### Plot predicted data points
    ax.plot(subsurface_data.ts.values[start_index:],np.concatenate((pred_train,pred_test)),lw = 1.5,label = 'Predicted')
    
    #### Plot predicted boundary
    ax.axvline(subsurface_data.ts.values[start_index + len(train_y)],ls = '--',color = tableau20[14])
    
    #### Plot legend
    ax.legend(bbox_to_anchor = (1.22,1.0),fontsize = 14)
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 16)
    ax.set_ylabel('Displacement (m)',fontsize = 16)
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #######################################################
    #### Plot cumulative displacement data predictions ####
    #######################################################
    
    #### Get smoothened and resampled displacement data
    subsurface_data_xz = subsurface_data[['ts','xz']].resample(ave_window,label = 'right',closed = 'right',on = 'ts').mean().rolling(smooth_window).mean()
    
    #### Get displacement column
    subsurface_data_xz['disp'] = subsurface_data_xz.xz - subsurface_data_xz.xz.shift(1)
    displacement = subsurface_data_xz.disp.dropna().values

    #### Denormalize quantities
    dn_actual = np.concatenate((train_y,test_y))*(max(displacement) - min(displacement)) + min(displacement)
    dn_pred = np.concatenate((pred_train,pred_test))*(max(displacement) - min(displacement)) + min(displacement)
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot measured data points
    ax.plot(subsurface_data.ts.values[start_index:],np.cumsum(dn_actual),label = 'Measured',lw = 3.0)
    
    #### Plot predicted data points
    ax.plot(subsurface_data.ts.values[start_index:],np.cumsum(dn_pred),lw = 1.5,label = 'Predicted')
    
    #### Plot predicted boundary
    ax.axvline(subsurface_data.ts.values[start_index + len(train_y)],ls = '--',color = tableau20[14])
    
    #### Plot legend
    ax.legend(bbox_to_anchor = (1.22,1.0),fontsize = 14)
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 16)
    ax.set_ylabel('Displacement (m)',fontsize = 16)
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #### Compute the rmse
    rmse = np.sqrt(np.sum(np.power(test_y - pred_test,2))/float(len(test_y)))
    print 'RMSE = ' , rmse
    
    return rmse

if __name__ == '__main__':
    
    subsurface_data = GetSubsurfaceData('2016-01-01', '2017-03-01', 'magta', 15)
 #   subsurface_data = GetDisplacementPlots(subsurface_data)
    train_x, train_y, test_x, test_y, X, Y = GetFeatureVectors2(subsurface_data, 4,
                                                         train_ratio=0.75)
    max_clf = GetOptimalCLF2(train_x,train_y,rand_starts = 8)
    
    rmse = GetTrainTestPlotResult2(subsurface_data, max_clf, train_x,
                                  train_y, test_x, test_y)
#    
#    subsurface_data2 = GetSubsurfaceData('2017-03-02', '2018-09-01', 'magta', 15)
#
#    train_x2, train_y2, test_x2, test_y2 = GetFeatureVectors2(subsurface_data2, 2,
#                                                         train_ratio=0.75)
#
#    max_clf.fit(train_x2,train_y2)    
#    
#    rmse = GetTrainTestPlotResult2(subsurface_data2, max_clf, train_x2,
#                                  train_y2, test_x2, test_y2)
