# -*- coding: utf-8 -*-
"""
Winnie Leung

Template QCB Analysis Workflow code
This script imports functions from qcbplotting.py.

Workflow includes:
    
    Import dataset from dsdb or from csv
    Getting overal count of subset structure groups
    Initial exploration/visualization of data with LDA/PCA (with pre-processing)
    Table of correlation coefficient to weight original features to LDA/PCA components
    Table of stats with mean and std of chosen features
    Scatter plots with 95% confidence interval (option to separate by session number)
    2D Scatter plots with linear regression (ODR) with R^2 and slope values (option to separate by session number)
    3D Scatter plots
    
"""
# %% Import Section

from pathlib import Path
import os

# uncomment if loading dataset from AICS databasedataset
#import datasetdatabase as dsdb

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

os.chdir(r'\\allen\aics\microscopy\Winnie\QCB\Python Scripts')

import qcbplotting as qp

# %% LOAD DATASET from Network
# connect to database(prod)
prod = dsdb.DatasetDatabase(config="//allen/aics/assay-dev/Analysis/QCB_database/prod_config.json")

# load features from golgi
ds_meta = prod.get_dataset(name='QCB_drug_cell_meta')
ds_dna_fea = prod.get_dataset(name='QCB_DRUG_DNA_feature')
ds_cell_fea = prod.get_dataset(name='QCB_DRUG_CELL_feature')
ds_gol_fea = prod.get_dataset(name='QCB_DRUG_ST6GAL_feature')
ds_tub_fea = prod.get_dataset(name='QCB_DRUG_TUBA1B_feature')
ds_sec_fea = prod.get_dataset(name='QCB_DRUG_SEC61B_feature')
ds_actb_fea = prod.get_dataset(name='QCB_DRUG_ACTB_feature')
ds_tjp1_fea = prod.get_dataset(name='QCB_DRUG_TJP1_feature')
ds_myo_fea = prod.get_dataset(name='QCB_DRUG_MYH10_feature')
ds_lamp1_fea = prod.get_dataset(name='QCB_DRUG_LAMP1_feature')

# %% Making dataframe with all features and meta data and save to csv

# Concatenate structure features table
# struc_fea = pd.concat(struc_dfs.values(), axis = 0)

struc_fea = pd.concat([ds_gol_fea.ds,
                       ds_tub_fea.ds,
                       ds_sec_fea.ds,
                       ds_actb_fea.ds,
                       ds_tjp1_fea.ds,
                       ds_myo_fea.ds,
                       ds_lamp1_fea.ds], axis=0)

# Inner join between dna and cell features
df_dna_cell_merge = pd.merge(ds_dna_fea.ds,
                             ds_cell_fea.ds,
                             on='cell_id', how='inner')

# Merge with dna/cell/struc_fea
df_allfea_merge = pd.merge(df_dna_cell_merge,
                           struc_fea,
                           on='cell_id', how='inner')

# Merge meta with all features
df_merge = pd.merge(df_allfea_merge,
                    ds_meta.ds,
                    on='cell_id', how='inner')

# Coerce it to dataframe
df = pd.DataFrame(df_merge)

# Save to network
export_dir = Path(r'\\allen\aics\microscopy\Winnie\QCB\Python Scripts\Drug datasets export')

df.to_csv(export_dir / 'df.csv', index=False)
ds_dna_fea.ds.to_csv(export_dir / 'ds_dna_fea.csv', index=False)
ds_cell_fea.ds.to_csv(export_dir / 'ds_cell_fea.csv', index=False)

ds_gol_fea.ds.to_csv(export_dir / 'ds_gol_fea.csv', index=False)
ds_tub_fea.ds.to_csv(export_dir / 'ds_tub_fea.csv', index=False)
ds_sec_fea.ds.to_csv(export_dir / 'ds_sec_fea.csv', index=False)
ds_actb_fea.ds.to_csv(export_dir / 'ds_actb_fea.csv', index=False)
ds_tjp1_fea.ds.to_csv(export_dir / 'ds_tjp1_fea.csv', index=False)
ds_myo_fea.ds.to_csv(export_dir / 'ds_myo_fea.csv', index=False)
ds_lamp1_fea.ds.to_csv(export_dir / 'ds_lamp1_fea.csv', index=False)

# %% Import files from csv's
import_dir = Path(r'\\allen\aics\microscopy\Winnie\QCB\Python Scripts\Drug datasets export')

df = pd.read_csv(import_dir / 'df.csv')
ds_dna_fea = pd.read_csv(import_dir / 'ds_dna_fea.csv')
ds_cell_fea = pd.read_csv(import_dir / 'ds_cell_fea.csv')

# store structure features into dictionary grouped by each structure
struc_dfs = {}
fea_csvs = [csv for csv in import_dir.glob('*fea.csv')
            if 'dna' not in csv.name and 'cell' not in csv.name]

for x in fea_csvs:
    struc_dfs[x.stem] = pd.read_csv(import_dir / x)

# %% GLOBAL VARIABLES

STRUCTURE = 'golgi'

# nominal columns
NOM_COLS = ['drug_label', 'cell_id', 'cell_ver', 'czi_filename',
            'idx_in_stack', 'roi', 'str_ver', 'structure_name']

# Features of interest
# dictionary storing list of feature column title grouped by each structure
STRUC_FEA_DIC = {}
for key, data in struc_dfs.items():
    STRUC_FEA_DIC[f'{key}'] = list(data)

STRUC_MAP = {'golgi': 'ds_gol_fea',
             'tubulin': 'ds_tub_fea',
             'sec61b': 'ds_sec_fea',
             'beta-actin': 'ds_actb_fea',
             'zo1': 'ds_tjp1_fea)',
             'myosin': 'ds_myo_fea',
             'lamp1': 'ds_lamp1_fea'}

DNA_FOI = list(ds_dna_fea)
DNA_FOI.remove('cell_id')

CELL_FOI = list(ds_cell_fea)
CELL_FOI.remove('cell_id')

STRUC_FOI = STRUC_FEA_DIC.get(STRUC_MAP.get(STRUCTURE))
STRUC_FOI.remove('cell_id')

ALL_FOI = DNA_FOI + CELL_FOI + STRUC_FOI

# Which FOI list to use
FOI = ALL_FOI

# %% CREATE SUBSETS AND COUNT

per_struct_drug = {}
for struct_name, group in df.groupby('structure_name'):
    drug_dict = {}
    for drug, subgroup in group.groupby('drug_label'):
        drug_dict[drug] = subgroup
    per_struct_drug[struct_name] = drug_dict

# Count how many cells per condition
counts_table = qp.get_condition_counts(df, 'structure_name', 'drug_label')

# Check if each drug treatment for golgi group has sufficient data
struc_count = counts_table[(counts_table.structure_name == STRUCTURE)]

# %%    ########  VISUALIZING DATA ############

# %matplotlib auto

# Make subsets, color mapping
struc_subset = df.groupby(by='structure_name').get_group(STRUCTURE)
# Save out nominal columns
nom_cols = struc_subset[NOM_COLS]

# fill struc_subset with DNA, CELL, and STR features
struc_subset = struc_subset[ALL_FOI]

# get color mapping of drug to color
# Change the drug_label column to category dtypes and change to codes
struc_subset['drug_label'] = nom_cols['drug_label']
struc_subset['drug_label'] = struc_subset['drug_label'].astype("category")
mapping = dict(enumerate(struc_subset['drug_label'].cat.categories))
mapping = list(mapping.values())
color_map = ['b', 'g', 'k', 'y', 'r', 'm', 'c']

# Add column for session number

session_number = []
for filename in nom_cols['czi_filename']:
    before, after = filename.split('-Scene-')
    session = before.split('_')[-1]
    session_number.append(session)

struc_subset['session_number'] = session_number
struc_subset['session_number'] = struc_subset['session_number'].astype('int64')

# assign color per drug
color_selection = [color_map[index] for index in range(len(mapping))]

# %% EXPORT ALL FOI

# noinspection PyPackageRequirements
export_dir = Path(r'C:\Users\Winnie\Desktop\testing')
exp_df = pd.DataFrame(STRUC_FOI)
exp_df.to_csv(export_dir / f'{STRUCTURE}_STRUC_FOI.csv',
              header=False, index=False)

# %% CUSTOMIZE FOI

# Customized foi, add all foi's to dictionary d
"""
# sec61b
foi = ['str_1st_axis_length_mean',
        'str_2nd_axis_length_mean',
        'str_3rd_axis_length_mean',
        'str_equator_eccentricity_mean',
        'str_surface_area_mean',
        'str_volume_mean',
        'str_meridional_eccentricity_mean',
        'str_number_of_components',
        'str_skeleton_edge_vol_mean',
        'str_skeleton_vol_mean',
        'str_sphericity_mean']
"""
# golgi
foi = ['str_1st_axis_length_mean',
       'str_2nd_axis_length_mean',
       'str_3rd_axis_length_mean',
       'str_equator_eccentricity_mean',
       'str_meridional_eccentricity_mean',
       'str_number_of_components',
       'str_sphericity_mean',
       'str_surface_area_mean',
       'str_volume_mean']
"""
# tubulin
foi = ['str_1st_axis_length_mean',
        'str_2nd_axis_length_mean',
        'str_3rd_axis_length_mean',
        'str_equator_eccentricity_mean',
        'str_meridional_eccentricity_mean',
        'str_number_of_components',
        'str_sphericity_mean',
        'str_skeleton_edge_vol_mean',
        'str_skeleton_prop_deg0_mean',
        'str_skeleton_prop_deg1_mean',
        'str_skeleton_prop_deg3_mean',
        'str_skeleton_prop_deg4p_mean',
        'str_skeleton_vol_mean',
        'str_surface_area_mean',
        'str_volume_mean']
"""
# Dictionary of feature grouping
d = {'DNA Features': DNA_FOI,
     'Cell Features': CELL_FOI,
     'DNA and Cell Features': DNA_FOI + CELL_FOI,
     'Structure Features': STRUC_FOI,
     'All Features': ALL_FOI,
     'Selected Structure Features': foi}

# %% Formatting df to run before LDA

# Turn NaN values into 0's (with number of structure components = 0 and 1)
struc_subset_filled = struc_subset[ALL_FOI]
struc_subset_filled = struc_subset_filled.fillna(0, inplace=False)

# Add drug_label column - LDA will ignore this column
struc_subset_filled['drug_label'] = nom_cols['drug_label']

# %% LDA - Linear Discriminant Analysis
# run fillna block first

# Change savegraphs to True to save graphs to export_dir
savegraphs = False
export_dir = Path(r'C:\Users\Winnie\Desktop\testing')

LDA_results = {}
sortby = 'Abs(C1)'

"""
# Example code for getting dataframe subset with only vehicle and one other 
# drug for finer analysis to find features that separate the two graphs

other_drug = 'Brefeldin'

struc_subset_filled = struc_subset_filled[(struc_subset_filled.drug_label == other_drug) |
                        (struc_subset_filled.drug_label == 'Vehicle')]
"""

for key, foi in d.items():
    lda_df = struc_subset_filled[foi]
    lda = LDA(n_components=3)
    T = lda.fit_transform(lda_df, y=struc_subset_filled['drug_label'])

    T_lab = qp.plot_T(T, color_selection, STRUCTURE, mapping, graphtype='LDA',
                  addtotitle = f'{key}',
                  drug_lab = struc_subset_filled['drug_label'],
                  savegraphs=savegraphs, savedir=export_dir)

    Corr_Table = qp.get_corr_table(struc_subset_filled[foi], T_lab, sort_by=sortby)
    exp_var_ratio = lda.explained_variance_ratio_
    weights = lda.coef_
    LDA_results.update({f'{key}': {'T_lab': T_lab,
                                   'exp_var_ratio': exp_var_ratio,
                                   'Weights': weights,
                                   'Corr_Table': Corr_Table}})

# %% PCA
# Graphing PCA by above feature categories and getting tables
# run fillna block first

savegraphs=False
export_dir = Path(r'C:\Users\Winnie\Desktop\testing')

PCA_results = {}
sort_by = 'Abs(C1)'
for key, foi in d.items():
    struc_subset_copy = struc_subset_filled[foi]
    struc_scaled = qp.scale_features_df(struc_subset_copy)  # Scale features for pre-processing for PCA
    pca = PCA(n_components=3, svd_solver="randomized")
    pca.fit(struc_scaled)
    T = pca.transform(struc_scaled)

    # Graph PCA and get correlation table of original features to each PC
    T_lab = qp.plot_T(T, color_selection, STRUCTURE, mapping, graphtype='PCA',
                  addtotitle = f'{key}',
                  drug_lab = struc_subset_filled['drug_label'],
                  savegraphs=savegraphs, savedir=export_dir)

    Corr_Table = qp.get_corr_table(struc_scaled[foi], T_lab, sort_by=sort_by)

    pca_expl_var = pca.explained_variance_ratio_
    PCA_results.update({f'{key}': {'T_lab': T_lab,
                                   'pca_expl_var': pca_expl_var,
                                   'Corr_Table': Corr_Table}})

# %% Get foi from top LDA features
# get top foi to do analysis and plot later

sort_by = 'Abs(C1)'
top = 10

table = LDA_results['All Features']['Corr_Table']
foi = table.sort_values(by=[sort_by], ascending=False).head(top)['Feature']

# %% STATISTICS: RETURNS PARAMETERS_TABLE WITH MEAN AND STD OF TOP FOI ACROSSS DRUG GROUPS

# data table with foi
stats_df = struc_subset
stats_df['drug_label'] = nom_cols['drug_label']

foi_stats = {}
for feature in foi:
    table = stats_df.groupby(['drug_label'])[feature].describe()
    foi_stats[feature] = table

# Compile list of feature parameter into 1 table
parameters = ['mean', 'std']  # what you want to calculate
parameters_table = {}
indices = foi_stats[list(foi_stats.keys())[0]].index.values.tolist()  # list of drug indices

# Assign True if you want to export parameters_table to csv's
export = True
export_dir = Path(r'C:\Users\Winnie\Desktop\testing')

# List of drugs into plot_order with Vehicle being first
plot_order = qp.get_plot_order(stats_df, 'Vehicle')

for par in parameters:
    parameter_table = {}
    parameter_table = pd.DataFrame(index=indices)

    for feature, stats in foi_stats.items():
        parameter_table[feature] = stats[par]

    parameter_table = parameter_table.transpose()
    parameter_table = parameter_table[plot_order]  # re-order columns

    parameters_table.update({f'{par}': parameter_table})

    if export:
        exp_df = pd.DataFrame(parameters_table[par])
        exp_df.to_csv(export_dir / f'par_table_{par}.csv',
                      index=indices)

# %% PLOTTING SCATTER CI PLOTS 

# Change savegraphs to True to save graphs to export_dir
savegraphs = False
export_dir = Path(r'C:\Users\Winnie\Desktop\testing')

scatter_df = struc_subset
scatter_df['drug_label'] = nom_cols['drug_label']

qp.plot_scatter_ci(scatter_df, STRUCTURE, foi, savegraphs=savegraphs,
                 savedir=export_dir)


# %% PLOTTING BY SESSION NUMBER

# Change savegraphs to True to save graphs to export_dir
savegraphs = False
export_dir = Path(r'C:\Users\Winnie\Desktop\testing')

# one plot per foi against session number
# foi = ['str_number_of_components', 'str_volume_mean', 'str_volume_total']
foi = ['str_volume_sum']
struc_subset['drug_label'] = nom_cols['drug_label']
drugs = qp.get_plot_order(struc_subset, 'Vehicle')

time_sep = 'session_number'


for drug in drugs:
    # Make subset of drug dataset
    scatter_df = struc_subset[(struc_subset.drug_label == drug)]

    plot_order = list(sorted(set(scatter_df[time_sep])))
    
    qp.plot_scatter_ci(scatter_df, STRUCTURE, foi, groupby=time_sep, 
                     addtotitle=f'{drug}', plot_order=plot_order, 
                     savegraphs=savegraphs, savedir=export_dir)

# %% Plot all Vehicle together then each drug by session

# Change savegraphs to True to save graphs to export_dir
savegraphs = False
# directory to save output graphs to
export_dir = Path(r'C:\Users\Winnie\Desktop\testing')

foi = ['str_volume_sum']
struc_subset['drug_label'] = nom_cols['drug_label']
drugs = qp.get_plot_order(struc_subset, 'Vehicle')

# directory to save output graphs to
export_dir = Path(r'C:\Users\Winnie\Desktop\testing')

time_sep = 'session_number'

for drug in drugs:
    # make subset of dataframe of all vehicle + drug
    dff = struc_subset[(struc_subset.drug_label == drug) | (struc_subset.drug_label == 'Vehicle')]

    dff.loc[dff.drug_label == 'Vehicle', 'session_number'] = 'Vehicle'
    # Make subset of brefeldin dataset
    scatter_df = struc_subset[(struc_subset.drug_label == drug)]

    plot_order = list(sorted(set(scatter_df[time_sep])))

    # Insert "Vehicle" to first element in list
    plot_order.insert(0, 'Vehicle')

    qp.plot_scatter_ci(dff, STRUCTURE, foi, groupby=time_sep, 
                     addtotitle=f'{drug}', plot_order=plot_order, 
                     savegraphs=savegraphs, savedir=export_dir)


# %% 2D Scatter plot with linear regression 
# with Orthogonal Distance Regression option

# Change savegraphs to True to save graphs to export_dir
savegraphs = False
export_dir = Path(r'C:\Users\Winnie\Desktop\testing')

scatter_df = struc_subset
scatter_df['drug_label'] = nom_cols['drug_label']
drugs = ['Vehicle', 'Brefeldin']

plot_foi = ['str_number_of_components', 'str_volume_mean', 'str_volume_sum']
x_lab = ['dna_volume', 'cell_volume']


# For plotting 2D scatter plots for each plot_foi per drug in drugs against
# each x_lab
for axis in x_lab:

    qp.plot_scatter_2d(scatter_df, STRUCTURE, mapping, color_selection, 
                     plot_foi, drugs, x_lab=axis, odrreg=True,
                     addtotitle=f'{axis}', savegraphs=savegraphs,
                     savedir=export_dir)


# %% 2D Scatter plot with linear regression separated by session number

# Change savegraphs to True to save graphs to export_dir
savegraphs = False
# directory to save output graphs to
export_dir = Path(r'C:\Users\Winnie\Desktop\testing')

# For plotting above plots separated by session number
group_sep = 'session_number'

scatter_df = struc_subset
scatter_df['drug_label'] = nom_cols['drug_label']

drugs = ['Vehicle', 'Brefeldin']
plot_foi = ['str_number_of_components']
x_lab = ['dna_volume']

for drug in drugs:
    # dataframe subset with current drug
    drug_subset = scatter_df.groupby('drug_label').get_group(drug)

    for session in list(sorted(set(drug_subset[group_sep]))):
        # dataframe subset with current drug and session
        dff = drug_subset.groupby(group_sep).get_group(session)

        for axis in x_lab:

            qp.plot_scatter_2d(dff, STRUCTURE, mapping, color_selection, 
                         plot_foi, drug, x_lab=axis, odrreg=True, 
                         addtotitle=f'{axis}__Session{session}',
                         savegraphs=savegraphs, savedir=export_dir)

# %% 3D Scatter plots                                                           # features to plot

# Change savegraphs to True to save graphs to export_dir
savegraphs = False
export_dir = Path(r'C:\Users\Winnie\Desktop\testing')

# 3D scatter plots over DNA volume and cell volume
plot_foi = foi

qp.plot_scatter_3d(scatter_df, STRUCTURE, mapping, color_selection, plot_foi,
                 savegraphs=savegraphs, savedir=export_dir)


# %% 
# ############  Non-workflow sample Code  #############


# Cell volume and dna volume analysis from different drugs

# first step: make sure you can aggregate the different structures
# plot cell_volume, DNA_volume, scaled cell_volume/dnavolume against drug

import_dir = Path(r'\\allen\aics\microscopy\Winnie\QCB\Python Scripts\Drug datasets export')

df = pd.read_csv(import_dir / 'df.csv', header=0)

# check you have all structures - get count
per_struct_drug = {}
for struct_name, group in df.groupby('structure_name'):
    drug_dict = {}
    for drug, subgroup in group.groupby('drug_label'):
        drug_dict[drug] = subgroup
    per_struct_drug[struct_name] = drug_dict

# Count how many cells per condition
counts_table = qp.get_condition_counts(df, 'structure_name')

dff = df
dff = dff.fillna(0, inplace=False)
dff['cell_to_dna_volume'] = dff['cell_volume'] / dff['dna_volume']
features = ['dna_volume', 'cell_volume', 'cell_to_dna_volume']
groupby = 'structure_name'

plot_order = qp.get_plot_order(dff, control='golgi', by='structure_name')

qp.plot_scatter_ci(dff, STRUCTURE, features, plotallstruc=True,
                   plot_order=plot_order, groupby=groupby)

# Need to remove Sec61b's outlier
sec61b = dff.groupby('structure_name').get_group('sec61b')
outlier_val = sec61b['dna_volume'].max()
dff.drop(dff[(dff.dna_volume == outlier_val)].index, inplace=True)

qp.plot_scatter_ci(dff, STRUCTURE, features, plotallstruc=True,
                   plot_order=plot_order, groupby=groupby)
