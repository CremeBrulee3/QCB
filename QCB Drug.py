# -*- coding: utf-8 -*-
"""
Winnie Leung

QCB Analysis Codes
Functions include:
    Get count tables for each condition
    PCA plots with Correlation coefficients table
    Parallel Coordinates
    Comparing variance of Control group vs Control + Drug
    Violin Distribution plots with quartiles and 95% CI
    
"""
# %% Import Section
import datasetdatabase as dsdb
import pandas as pd
from itkwidgets import view
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import aicsimageio
from pandas.plotting import parallel_coordinates
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import manifold

## ggplot
plt.style.use('ggplot')

# %% LOADING DATASETS

## Code from Jianxu

## connect to database(prod)
prod = dsdb.DatasetDatabase(config="//allen/aics/assay-dev/Analysis/QCB_database/prod_config.json")

## load meta table, the table of all current features
ds_meta = prod.get_dataset(name='QCB_drug_cell_meta')
ds_fea = prod.get_dataset(name='QCB_drug_fea_v0')

## Merge tables
df_merge = pd.merge(ds_meta.ds, ds_fea.ds, on='cell_id', how='inner')

## Coerce it to dataframe
df = pd.DataFrame(df_merge)


# %% FUNCTIONS
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

## return list of colors and legends for graphing
def ColorAndLegend(df, mapping):
    ## set different colors for each group
    colors = []
    legend = []
    color_selection = ['b', 'g', 'k', 'y', 'r']
    legend_selection = []
    for x in range(0, len(mapping)):
        legend_selection.append(mapping[x])
    
    ## Legend = list of drug_label names of input data
    for cat in df['drug_label']:
        colors.append(color_selection[cat])
        legend.append(legend_selection[cat])
    
    return colors, legend
 
## Calculate correlation coefficient between every column of struc_subset 
## and a column vector of PCA_T. Returns table of correlation coeffs
def GetCorrCoeff(struc_subset, PCA_T):                                          ## PCA_T is 1-D list of one of PC
    corr_coeff = []
    col_list = list(struc_subset)
    for col in col_list:    
        corr = np.corrcoef(struc_subset[col], PCA_T)[1,0]                       ## just corrcoeff returns a 2x2 matrix (diagonal 1's)
        corr_coeff.append(corr)
    return corr_coeff


## Do PCA on scaled whole dataset. Returns exp_var_ratio and matrix T
## Principal Component Analysis
def GraphPCA(struc_scaled, mapping, color_selection, addtotitle=None):
    pca = PCA(n_components = 3, svd_solver = "randomized")
    pca.fit(struc_scaled)
    T = pca.transform(struc_scaled)
        
    T_lab = pd.DataFrame({
            'drug_label': nom_cols['drug_label'],
            'C1': T[:,0],
            'C2': T[:,1],
            'C3': T[:,2]})
            
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    ax.set_title('PCA of {} {}'.format(STRUCTURE, addtotitle))
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    
    for index in mapping:
        drug = mapping[index]
        xyz = T_lab.groupby('drug_label').get_group(drug)
        color = color_selection[index]
        ax.scatter(xyz['C1'], xyz['C2'], xyz['C3'], 
                   c=color, label=drug)
    
    ## Plot
    plt.legend()
    plt.show()

    ## Percentage of variance explained by each of the selected components
    pca_expl_var = pca.explained_variance_ratio_
    
    return T_lab, pca_expl_var

## Find the correlation coefficient for all three PC's. Returns corr table
def GetCorrTable(struc_scaled, T_lab, sort_by='Abs(C2)'):
    C1_corr = GetCorrCoeff(struc_scaled, T_lab['C1'])                         ## same correlation using original scaled or unscaled
    C2_corr = GetCorrCoeff(struc_scaled, T_lab['C2'])
    C3_corr = GetCorrCoeff(struc_scaled, T_lab['C3'])
    
    corr_table = pd.DataFrame({'Feature': list(struc_scaled),
                               'Abs(C1)': [abs(x) for x in C1_corr],
                               'Abs(C2)': [abs(y) for y in C2_corr],
                               'Abs(C3)': [abs(z) for z in C3_corr],
                               'C1': C1_corr,
                               'C2': C2_corr,
                               'C3': C3_corr})
    
    corr_table_sorted = corr_table.sort_values(by = [sort_by], 
                                               ascending = False)
    return corr_table_sorted

## Evaluating changes in data variance 
## between vehicle group and with drug group. Returns table fo variances 
def CompareVar(struc_subset, drug):
    vehicle = struc_subset.groupby(by = 'drug_label').get_group('Vehicle')
    drug_gp = struc_subset.groupby(by = 'drug_label').get_group(drug)
    compare_drug = pd.concat([vehicle, drug_gp])

    veh_var = vehicle.var(axis = 0)                                             ## automatically dropped nominal columns
    comp_drug_var = compare_drug.var(axis = 0)
    ## Evaluating absolute variance changes (not scaled)
    var_comparison = pd.DataFrame({'Vehicle': veh_var,
                                   'Vehicle and {}'.format(drug): 
                                       comp_drug_var,
                                   'Difference': comp_drug_var - veh_var,
                                   'Percent Variance Change':
                                       (comp_drug_var - veh_var)/veh_var*100})

    ## drop drug label variance - irrelevant
    #var_comparison = var_comparison.drop(['drug_label'])
    
    ## sorting from higest change to lowest
    var_comparison_sorted = var_comparison.sort_values(by = 
                                                       ['Percent Variance Change'],
                                                       ascending = False)

    return var_comparison_sorted

## Plot Violin plots of different drugs
## df = data, features = list of features, # of graphs
def PlotViolin(df, features, plot_order):
    # Set up the matplotlib figure
    sns.set(style="darkgrid")
    for feature in features:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Draw a violinplot
        ax = sns.violinplot(data=df, x="drug_label", y=feature, 
                       palette="Pastel1", order=plot_order)

def PlotScatterBox(df, features, plot_order):
    # Set up the matplotlib figure
    sns.set(style="darkgrid")
    for feature in features:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        
        ax = sns.stripplot(x='drug_label', y=feature, data=df, jitter=True,
                           palette="Pastel1", order=plot_order)
        # Draw a boxplot
        ax = sns.boxplot(x='drug_label', y=feature, data=df, order=plot_order)
        
        
## 3D scatter plot: dff = dataset, plot_foi = list of features to plot
## x and y_lab are feature names on x and y axis
def PlotScatter(dff, plot_foi, x_lab='dna_volume', y_lab='mem_volume'):
    for foi in plot_foi:
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        z_lab = foi
    
        ax.set_title('{}: {}'.format(STRUCTURE, foi))
        ax.set_xlabel('{}'.format(x_lab))
        ax.set_ylabel('{}'.format(y_lab))
        ax.set_zlabel('{}'.format(z_lab))
        
        for index in mapping:
            drug = mapping[index]
            drug_group = dff.groupby('drug_label').get_group(drug)
            color = color_selection[index]
            ax.scatter(drug_group[x_lab], drug_group[y_lab], drug_group[z_lab], 
                       c=color, label=drug)
        
        ## Plot
        plt.legend()
        plt.show()    
        

def PlotIsomap(iso_df, addtotitle=None):
    iso = manifold.Isomap(n_neighbors=4, n_components=3)
    T = iso.fit_transform(iso_df)
    
    T_lab = pd.DataFrame({
            'drug_label': nom_cols['drug_label'],
            'C1': T[:,0],
            'C2': T[:,1],
            'C3': T[:,2]})
            
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    ax.set_title('Isomap of {} {}'.format(STRUCTURE, addtotitle))
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    
    for index in mapping:
        drug = mapping[index]
        xyz = T_lab.groupby('drug_label').get_group(drug)
        color = color_selection[index]
        ax.scatter(xyz['C1'], xyz['C2'], xyz['C3'], 
                   c=color, label=drug, alpha=0.75)
    
    ## Plot
    plt.legend()
    plt.show()
    
    return T_lab

# %% GLOBAL VARIABLES
    
STRUCTURE = 'golgi'
NOM_COLS = ['drug_label', 'cell_id', 'cell_ver', 'czi_filename', 
                   'idx_in_stack', 'roi', 'str_ver', 'structure_name']

## Features of interest

ALL_FOI = ['dna_1st_axis_length',
     'dna_1st_eigenvalue',
     'dna_2nd_axis_length',
     'dna_2nd_eigenvalue',
     'dna_3rd_axis_length',
     'dna_3rd_eigenvalue',
     'dna_equator_eccentricity',
     'dna_meridional_eccentricity',
     'dna_sphericity',
     'dna_surface_area',
     'dna_volume',
     'mem_1st_axis_length',
     'mem_1st_eigenvalue',
     'mem_2nd_axis_length',
     'mem_2nd_eigenvalue',
     'mem_3rd_axis_length',
     'mem_3rd_eigenvalue',
     'mem_equator_eccentricity',
     'mem_meridional_eccentricity',
     'mem_sphericity',
     'mem_surface_area',
     'mem_volume',
     'structure_1st_axis_length',
     'structure_1st_eigenvalue',
     'structure_2nd_axis_length',
     'structure_2nd_eigenvalue',
     'structure_3rd_axis_length',
     'structure_3rd_eigenvalue',
     'structure_equator_eccentricity',
     'structure_meridional_eccentricity',
     'structure_sphericity',
     'structure_surface_area',
     'structure_volume']

DNA_FOI = ['dna_1st_axis_length',
           'dna_1st_eigenvalue',
           'dna_2nd_axis_length',
           'dna_2nd_eigenvalue',
           'dna_3rd_axis_length',
           'dna_3rd_eigenvalue',
           'dna_equator_eccentricity',
           'dna_meridional_eccentricity',
           'dna_sphericity',
           'dna_surface_area',
           'dna_volume']


MEM_FOI = ['mem_1st_axis_length',
           'mem_1st_eigenvalue',
           'mem_2nd_axis_length',
           'mem_2nd_eigenvalue',
           'mem_3rd_axis_length',
           'mem_3rd_eigenvalue',
           'mem_equator_eccentricity',
           'mem_meridional_eccentricity',
           'mem_sphericity',
           'mem_surface_area',
           'mem_volume']

STRUC_FOI = ['structure_1st_axis_length',
             'structure_1st_eigenvalue',
             'structure_2nd_axis_length',
             'structure_2nd_eigenvalue',
             'structure_3rd_axis_length',
             'structure_3rd_eigenvalue',
             'structure_equator_eccentricity',
             'structure_meridional_eccentricity',
             'structure_sphericity',
             'structure_surface_area',
             'structure_volume']

## Which FOI list to use
FOI = STRUC_FOI

# %% CREATE SUBSETS AND COUNT
  
per_struct_drug = {}
for struct_name, group in df.groupby('structure_name'):
    drug_dict = {}
    for drug, subgroup in group.groupby('drug_label'):
        drug_dict[drug] = subgroup
    per_struct_drug[struct_name] = drug_dict
    
## Count how many cells per condition    
counts_table = GetConditionCounts('structure_name', 'drug_label')

## Check if each drug treatment for golgi group has sufficient data
struc_count = counts_table[(counts_table.structure_name == STRUCTURE)] 


# %% VISUALIZING DATA

## Scale all features and visualize by PCA

struc_subset = df.groupby(by='structure_name').get_group(STRUCTURE)
## Save out nominal columns
nom_cols = struc_subset[NOM_COLS]

## get color mapping of drug to color
## Change the drug_label column to category dtypes and change to codes
struc_subset['drug_label'] = struc_subset['drug_label'].astype("category")
mapping = dict(enumerate(struc_subset['drug_label'].cat.categories))
#struc_subset['drug_label'] = struc_subset['drug_label'].cat.codes
color_selection = ['b', 'g', 'k', 'y', 'r']

## Dictionary of feature grouping
d = {'DNA Features': DNA_FOI, 
     'MEM Features': MEM_FOI, 
     'Structure Features': STRUC_FOI,
     'All Features': ALL_FOI}

## Graphing PCA by above feature categories and getting tables
PCA_results={}
sort_by = 'Abs(C1)'
for key, foi in d.items():
    struc_subset_copy = struc_subset[foi]
    struc_scaled = scaleFeaturesDF(struc_subset_copy)                           ## Scale features for pre-processing for PCA

    ## Graph PCA and get correlation table of original features to each PC
    [T_lab, pca_expl_var] = GraphPCA(struc_scaled, mapping, color_selection,
                                        addtotitle = ': {}'.format(key))
    Corr_Table = GetCorrTable(struc_scaled, T_lab, sort_by = sort_by)
    PCA_results.update({'{}'.format(key): {'T_lab': T_lab,
                                            'pca_expl_var': pca_expl_var,
                                            'Corr_Table': Corr_Table}})

## Isomap for nonlinear dimensionality reduction
Iso_results = {}
sort_by = 'Abs(C2)'
for key, foi in d.items():
    iso_df = struc_subset[foi]                                                  ## Structure FOI's
    T_lab = PlotIsomap(iso_df, addtotitle = ': {}'.format(key))
    Iso_Corr_Table = GetCorrTable(struc_scaled, T_lab, sort_by = sortby)
    Iso_results.update({'{}'.format(key): {'Corr_Table': Iso_Corr_Table}})
    
# %% STATISTICS

## Preliminary stats
compare = struc_subset_copy
compare['drug_label'] = nom_cols['drug_label']                                  ## Need drug_label column to pass into CompareVar

## Get table comparing variance for each drug
variance_sorted = {}
for index in mapping:
    drug = mapping[index]
    variance_sorted.update({drug: CompareVar(compare, drug)})

## Student T-Test compare group to Vehicle group
## ignore for now, violin plot includes 95% Confidence interval

# %% PLOTTING DATA

## plot_order - select which drug groups to plot
plot_order = ['Vehicle']
for i in range(0, len(mapping)):
    if mapping[i] not in plot_order:
        plot_order.append(mapping[i])

plot_order = ['Vehicle',
              'Brefeldin',
              'Paclitaxol',
              'Staurosporine']                                          ## for plotting only these 2

## Violin Plot: density plot with quartiles & 95% CI                           
PlotViolin(struc_subset, STRUC_FOI, plot_order)
PlotScatterBox(struc_subset, STRUC_FOI, plot_order)
## 3D scatter plots over DNA volume and membrane volume
plot_foi = STRUC_FOI                                                            ## features to plot
PlotScatter(struc_subset, plot_foi)
#PlotScatter(struc_subset, ['structure_meridional_eccentricity'], y_lab='dna_meridional_eccentricity')

#############################################################################



#counts_table.to_csv("C:/Users/winniel/Desktop/counts.csv")
# Parallel Coordinates Start Here:
plt.figure()
parallel_coordinates(compare_drug_scaled, 'drug_label')
plt.show()