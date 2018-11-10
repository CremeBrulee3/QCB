# -*- coding: utf-8 -*-
"""
Winnie Leung

QCB Analysis Codes
Functions include:
    Get count tables for each condition
    PCA, IsoMap, LDA plots with Correlation coefficients table
    Table of stats with mean and std of each group
    Scatter plots with t-test and p-values
    2D Scatter plots with 95% CI
    3D Scatter plots
    
"""

import os
import pandas as pd

#from itkwidgets import view  ## for jupyter notebook
import numpy as np
from scipy import stats
from scipy.odr import *

import matplotlib.pyplot as plt

from pandas.plotting import parallel_coordinates
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import manifold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

## Method to count number of entries in each structure and drug, returns table 
def GetConditionCounts(*features):
    conditions = df.groupby(list(features)).size().reset_index(name='Count')
    return conditions

## Function for scaling original data's variance. Returns scaled dataset
## Pre-processing before doing PCA
def scaleFeaturesDF(df):
    scaled = preprocessing.StandardScaler().fit_transform(df)
    scaled = pd.DataFrame(scaled, columns=df.columns)
    
    print("New Variances:\n", scaled.var())
    print("New Describe:\n", scaled.describe())
    return scaled

 
## Calculate correlation coefficient between every column of struc_subset 
## and a column vector of PCA_T. Returns table of correlation coeffs
def GetCorrCoeff(struc_subset, PCA_T):                                          ## PCA_T is 1-D list of one of PC
    corr_coeff = []
    col_list = list(struc_subset)
    for col in col_list:    
        corr = np.corrcoef(struc_subset[col], PCA_T)[1,0]                       ## just corrcoeff returns a 2x2 matrix (diagonal 1's)
        corr_coeff.append(corr)
    return corr_coeff


## Plot 3D graphs of dimensionality reduction (PCA, LDA, Isomap)
def PlotT(T, color_selection, graphtype=None, addtotitle=None, drug_lab=None):
    dimensionality = T.shape[1]
    column_data = {f'C{i+1}': T[:, i] for i in range(dimensionality)}
    T_lab = pd.DataFrame({
            'drug_label': drug_lab,
            **column_data})                                                     ## ** for expands dictionary
            
    fig = plt.figure()
    projection = '3d' if dimensionality == 3 else None
    ax = fig.add_subplot(111, projection=projection)
    
    if addtotitle == None:
        ax.set_title(f'{graphtype} of {STRUCTURE}')
    else:
        ax.set_title(f'{graphtype} of {STRUCTURE}, {addtotitle}')
    ax.set_xlabel('Component 1')
    if dimensionality > 1:
        ax.set_ylabel('Component 2')
    if dimensionality > 2:
        ax.set_zlabel('Component 3')
    
    for drug in drug_lab.unique():
        xyz = T_lab.groupby('drug_label').get_group(drug)
        index = mapping.index(drug)
        color = color_selection[index]
        if dimensionality == 1:
            ax.hist(*[xyz[c].values for c in xyz.columns if c.startswith('C')],
                    label=drug, alpha=0.5)
            continue
        else: 
            ax.scatter(*[xyz[c].values for c in xyz.columns if c.startswith('C')], 
                       c=color, label=drug, alpha=0.75)
    
    ## Plot
    plt.legend()
    plt.show()
    
    return T_lab

## Find the correlation coefficient for all three PC's. Returns corr table
def GetCorrTable(struc_subset, T_lab, sort_by='Abs(C1)'):
    dimensionality = T_lab.shape[1] - 1
    
    C_table = {}
    C_table.update({'C1_corr': GetCorrCoeff(struc_subset, T_lab['C1'])})        ## same correlation using original scaled or unscaled
    if dimensionality > 1:
        C_table.update({'C2_corr': GetCorrCoeff(struc_subset, T_lab['C2'])})
    if dimensionality > 2:
        C_table.update({'C3_corr': GetCorrCoeff(struc_subset, T_lab['C3'])})
    
    corr_data_abs = {f'Abs(C{i+1})': [abs(x) for x in C_table[f'C{i+1}_corr']] for i in range(dimensionality)}
    corr_data = {f'C{i+1}': C_table[f'C{i+1}_corr'] for i in range(dimensionality)}
    
    corr_table = pd.DataFrame({'Feature': list(struc_subset),
                               **corr_data_abs,
                               **corr_data})    
    
    corr_table_sorted = corr_table.sort_values(by = [sort_by], 
                                               ascending = False)
    return corr_table_sorted


## Plot Violin plots of different drugs
## df = data, features = list of features, # of graphs
def PlotViolin(df, control, features, plot_order=""):
    
    if plot_order == "":
        plot_order = GetPlotOrder(df, control)
    
    # Set up the matplotlib figure
    sns.set(style="darkgrid")
    for feature in features:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Draw a violinplot
        ax = sns.violinplot(data=df, x="drug_label", y=feature, 
                       palette="Pastel1", order=plot_order)

def GetPlotOrder(df, control, by='drug_label'):
    plot_order = [control]
    for drug in list(df[by].unique()):
        if drug not in plot_order:
            plot_order.append(drug)
    
    return plot_order

## Plot Scatter plot for each category with 95% confidence interval
def PlotScatterCI(df, features, control='Vehicle', groupby='drug_label', 
                  addtotitle=None, plotallstruc=False, plot_order="",
                  savegraphs=False, savedir=''):
    
    if plot_order == "":
        plot_order = GetPlotOrder(df, control, by=groupby)
    
    # Set up the matplotlib figure
    sns.set(style="darkgrid")
    
    p_val_results = {}
    for feature in features:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        if addtotitle == None:
            if plotallstruc:
                ax.set_title(f'{feature}')
            else:
                STRUCTURE = df['structure_name'][0]
                ax.set_title(f'{STRUCTURE}: {feature}')
        else:
            if plotallstruc:
                ax.set_title(f'{feature}, {addtotitle}')
            else:
                STRUCTURE = df['structure_name'][0]
                ax.set_title(f'{STRUCTURE}: {feature}, {addtotitle}')
            
        ax = sns.pointplot(x=groupby, y=feature, data=df, ci=95, size=3,
                           color='k', join=False, order=plot_order,
                           estimator=np.mean)
        
        ## Make sure pointplot on top                                           
        plt.setp(ax.lines, zorder=100)
        plt.setp(ax.collections, zorder=100, label="")
        
        sns.stripplot(x=groupby, y=feature, data=df, jitter=True,
                          palette="Pastel1", edgecolor='k', size=5, 
                          order=plot_order)
        
        if savegraphs:
            fig.set_size_inches(10,6)
            fig.savefig(os.path.join(savedir, f'{feature}__{addtotitle}.png'))

## 2D scatter plot: dff = dataset, plot_foi = list of features to plot
## x and y_lab are feature names on x and y axis
def PlotScatter2D(dff, plot_foi, *doi, x_lab='dna_volume', linreg=False, 
                  odrreg=False, addtotitle=None):
    
    if isinstance(doi[0], list):
        doi = doi[0]
    
    STRUCTURE = df['structure_name'][0]
    
    for foi in plot_foi:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        y_lab = foi
    
        if addtotitle == None:
            ax.set_title(f'{STRUCTURE}: {foi}')
        else:
            ax.set_title(f'{STRUCTURE}: {foi}, {addtotitle}')
        ax.set_xlabel('{}'.format(x_lab))
        ax.set_ylabel('{}'.format(y_lab))
        
        for drug in doi:
            index = mapping.index(drug)
            try:
                drug_group = dff.groupby('drug_label').get_group(drug)
                color = color_selection[index]
                ax.scatter(drug_group[x_lab], drug_group[y_lab],
                           c=color, label=drug)
                
                x = drug_group[x_lab]
                y = drug_group[y_lab]
                
                x_norm = x/np.mean(x)
                y_norm = y/np.mean(y)
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_norm, y_norm)
                r_sq = r_value**2
                
                if linreg:
                    #ax.plot(x, (intercept + slope*x_norm)*np.mean(y), c='r', 
                            #label='$OLR: R^2$: {:.3f}'.format(r_sq))
                    pass
                
                if odrreg:
                    ## Model for fitting
                    linear_model = Model(linear_f)
                    
                    ## Real Data Object             
                    data = Data(x, y)
                    
                    ## Set up ODR with model and data
                    odr = ODR(data, linear_model, beta0=[0, 1])
                    odr.set_job(fit_type=0)
                    out = odr.run()
                    
                    ## Generate fitted data
                    #x = x.sort_values()
                    #x = list(x)
                    #x_fit = np.linspace(x[0], x[-1], 1000)
                    y_fit = linear_f(out.beta, x)
                    #y_fit = linear_f(out.beta, x_norm)
                    #ax.plot(x, y_fit*np.mean(y), c='k', label='ODR')
                    odrslope = out.beta[0]
                    ax.plot(x, y_fit, c='k', label=f'ODR. $R^2$: {r_sq:.3f}, slope={odrslope:.3f}')

            except:
                print(f'Skipped plotting {drug}')
                pass
            
        ## Plot
        plt.legend()
        plt.show()

## 3D scatter plot: dff = dataset, plot_foi = list of features to plot
## x and y_lab are feature names on x and y axis
def PlotScatter3D(dff, plot_foi, x_lab='dna_volume', y_lab='mem_volume'):
    
    STRUCTURE = df['structure_name'][0]
    for foi in plot_foi:
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        z_lab = foi
    
        ax.set_title(f'{STRUCTURE}: {foi}')
        ax.set_xlabel(f'{x_lab}')
        ax.set_ylabel(f'{y_lab}')
        ax.set_zlabel(f'{z_lab}')
        
        for drug in mapping:
            index = mapping.index(drug)
            try:
                drug_group = dff.groupby('drug_label').get_group(drug)
                color = color_selection[index]
                ax.scatter(drug_group[x_lab], drug_group[y_lab], 
                           drug_group[z_lab], c=color, label=drug)
            except:
                print('Skipped plotting {}'.format(drug))
                pass
        
        ## Plot
        plt.legend()
        plt.show()    

## Function to fit data with
def linear_f(p, x):
    m, c = p
    return m*x + c