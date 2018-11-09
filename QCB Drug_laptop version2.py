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
# %% Import Section

import sys
import os

os.chdir(r'\\allen\aics\microscopy\Winnie\QCB\Python Scripts')

import qcbplotting as qp

## for AICS databasedataset
import datasetdatabase as dsdb
## ggplot
#plt.style.use('ggplot')

# %% LOAD DATASET from Network 
## connect to database(prod)
prod = dsdb.DatasetDatabase(config="//allen/aics/assay-dev/Analysis/QCB_database/prod_config.json")

## load features from golgi
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


# %% Import files from csv's 
from pathlib import Path

import_dir = Path(r'\\allen\aics\microscopy\Winnie\QCB\Python Scripts\Drug datasets export')

df = pd.read_csv(import_dir / 'df.csv')
ds_dna_fea = pd.read_csv(import_dir / 'ds_dna_fea.csv')
ds_cell_fea = pd.read_csv(import_dir / 'ds_cell_fea.csv')

## structure features
struc_dfs = {}
fea_csvs = [csv for csv in import_dir.glob('*fea.csv') 
            if 'dna' not in csv.name and 'cell' not in csv.name]

for x in fea_csvs:
    struc_dfs[x.stem] = pd.read_csv(import_dir / x)
    
# %% Making dataframe with all features and meta data and save to csv

## Concatenate structure features table
    
#struc_fea = pd.concat(struc_dfs.values(), axis = 0)

struc_fea = pd.concat([ds_gol_fea.ds,
                     ds_tub_fea.ds,
                     ds_sec_fea.ds,
                     ds_actb_fea.ds,
                     ds_tjp1_fea.ds,
                     ds_myo_fea.ds,
                     ds_lamp1_fea.ds], axis = 0)

## Inner join between dna and cell features
df_dna_cell_merge = pd.merge(ds_dna_fea.ds, 
                            ds_cell_fea.ds,
                            on='cell_id', how='inner')

## Merge with dna/cell/struc_fea
df_allfea_merge = pd.merge(df_dna_cell_merge, 
                            struc_fea,
                            on='cell_id', how='inner')

## Merge meta with all features
df_merge = pd.merge(df_allfea_merge,
                    ds_meta.ds,
                    on='cell_id', how='inner')

## Coerce it to dataframe
df = pd.DataFrame(df_merge)

## Save to network

export_dir = r'\\allen\aics\microscopy\Winnie\QCB\Python Scripts\Drug datasets export'

df.to_csv(os.path.join(export_dir, 'df.csv'), index = False)
dff.to_csv(os.path.join(export_dir, 'dff.csv'), index = False)
ds_dna_fea.ds.to_csv(os.path.join(export_dir, 'ds_dna_fea.csv'), index = False)
ds_cell_fea.ds.to_csv(os.path.join(export_dir, 'ds_cell_fea.csv'), index = False)

ds_gol_fea.ds.to_csv(os.path.join(export_dir, 'ds_gol_fea.csv'), index = False)
ds_tub_fea.ds.to_csv(os.path.join(export_dir, 'ds_tub_fea.csv'), index = False)
ds_sec_fea.ds.to_csv(os.path.join(export_dir, 'ds_sec_fea.csv'), index = False)
ds_actb_fea.ds.to_csv(os.path.join(export_dir, 'ds_actb_fea.csv'), index = False)
ds_tjp1_fea.ds.to_csv(os.path.join(export_dir, 'ds_tjp1_fea.csv'), index = False)
ds_myo_fea.ds.to_csv(os.path.join(export_dir, 'ds_myo_fea.csv'), index = False)
ds_lamp1_fea.ds.to_csv(os.path.join(export_dir, 'ds_lamp1_fea.csv'), index = False)



# %% Cell volume and dna volume analysis from different drugs
    
## first step: make sure you can aggregate the different structures
## plot cell_volume, DNA_volume, scaled cell_volume/dnavolume against drug

import_dir = r'\\allen\aics\microscopy\Winnie\Scripts and Codes\Python Scripts\QCB\Drug datasets export'

df = pd.read_csv(os.path.join(import_dir, 'old_df.csv'), header = 0)


## check you have all structures - get count
per_struct_drug = {}
for struct_name, group in df.groupby('structure_name'):
    drug_dict = {}
    for drug, subgroup in group.groupby('drug_label'):
        drug_dict[drug] = subgroup
    per_struct_drug[struct_name] = drug_dict
    
## Count how many cells per condition    
counts_table = qp.GetConditionCounts('structure_name')

dff = df
dff = dff.fillna(0, inplace=False)
dff['cell_to_dna_volume'] = dff['cell_volume']/dff['dna_volume']
#dff['drug_label'] = dff['drug_label'].replace({'S-Nitro-Blebbistatin': 's-Nitro-Blebbistatin'})
features = ['dna_volume', 'cell_volume', 'cell_to_dna_volume']
groupby = 'structure_name'

plot_order = qp.GetPlotOrder(dff, 'Vehicle')

qp.PlotScatterCI(dff, features, groupby=groupby)

## Need to remove Sec61b's outlier
sec61b = dff.groupby('structure_name').get_group('sec61b')
outlier_val = sec61b['dna_volume'].max()
dff.drop(dff[(dff.dna_volume == outlier_val)].index, inplace=True)

qp.PlotScatterCI(dff, features, groupby=groupby)

## Perform pair-wise t-test/ anova
sec61b = dff.groupby('structure_name').get_group('sec61b')
betaactin = dff.groupby('structure_name').get_group('beta-actin')
golgi = dff.groupby('structure_name').get_group('golgi')
lamp1 = dff.groupby('structure_name').get_group('lamp1')
myosin = dff.groupby('structure_name').get_group('myosin')
tubulin = dff.groupby('structure_name').get_group('tubulin')
zo1 = dff.groupby('structure_name').get_group('zo1')

from scipy import stats
param = 'cell_to_dna_volume'
f_value, p_value = stats.f_oneway(sec61b[param], betaactin[param], 
                                  golgi[param], lamp1[param], myosin[param], 
                                  tubulin[param], zo1[param])


# %% Extract just DMSO

DMSO_subset = dff[(dff.drug_label == 'Vehicle')]

features = ['dna_volume', 'cell_volume', 'cell_to_dna_volume']
groupby = 'structure_name'
plot_order = list(DMSO_subset['structure_name'].unique())

qp.PlotScatterCI(DMSO_subset, features, groupby=groupby)

## Perform pair-wise t-test/ anova
sec61b = DMSO_subset.groupby('structure_name').get_group('sec61b')
betaactin = DMSO_subset.groupby('structure_name').get_group('beta-actin')
golgi = DMSO_subset.groupby('structure_name').get_group('golgi')
lamp1 = DMSO_subset.groupby('structure_name').get_group('lamp1')
myosin = DMSO_subset.groupby('structure_name').get_group('myosin')
tubulin = DMSO_subset.groupby('structure_name').get_group('tubulin')
zo1 = DMSO_subset.groupby('structure_name').get_group('zo1')

## Only gives 1 overall p-value for whole set of category
anova_test = {}
for par in features:
    f_value, p_value = stats.f_oneway(sec61b[par], betaactin[par], 
                                      golgi[par], lamp1[par], myosin[par], 
                                      tubulin[par], zo1[par])
    anova_test.update({f'{par}':{'f_value': f_value,
                                   'p_value': p_value}})



# %% GLOBAL VARIABLES
    
STRUCTURE = 'golgi'
NOM_COLS = ['drug_label', 'cell_id', 'cell_ver', 'czi_filename', 
            'idx_in_stack', 'roi', 'str_ver', 'structure_name']

## Features of interest

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

## Which FOI list to use
FOI = ALL_FOI

# %% CREATE SUBSETS AND COUNT
  
per_struct_drug = {}
for struct_name, group in df.groupby('structure_name'):
    drug_dict = {}
    for drug, subgroup in group.groupby('drug_label'):
        drug_dict[drug] = subgroup
    per_struct_drug[struct_name] = drug_dict
    
## Count how many cells per condition    
counts_table = qp.GetConditionCounts('structure_name', 'drug_label')

## Check if each drug treatment for golgi group has sufficient data
struc_count = counts_table[(counts_table.structure_name == STRUCTURE)] 


# %%    ###############  VISUALIZING DATA ########################

%matplotlib auto

## Make subsets, color mapping

struc_subset = df.groupby(by='structure_name').get_group(STRUCTURE)
## Save out nominal columns
nom_cols = struc_subset[NOM_COLS]

## fill struc_subset with DNA, CELL, and STR features
struc_subset = struc_subset[ALL_FOI]

"""
## list of columns that start with 'str' and ends with 'std'
str_std_list = list(one_comp.filter(regex='^str.*std$'))
str_list = list(one_comp.filter(regex='^str'))

"""
## get color mapping of drug to color
## Change the drug_label column to category dtypes and change to codes
struc_subset['drug_label'] = nom_cols['drug_label']
struc_subset['drug_label'] = struc_subset['drug_label'].astype("category")
mapping = dict(enumerate(struc_subset['drug_label'].cat.categories))
mapping = list(mapping.values())
color_map = ['b', 'g', 'k', 'y', 'r', 'm', 'c']

## Add column for scene number
scene_number = []
session_number = []
for filename in nom_cols['czi_filename']:
    before, after = filename.split('-Scene-')
    scene = after.split('-')[0]
    session = before.split('_')[-1]
    scene_number.append(scene)
    session_number.append(session)
    
struc_subset['scene_number'] = scene_number
struc_subset['session_number'] = session_number
struc_subset['scene_number'] = struc_subset['scene_number'].astype('int64')
struc_subset['session_number'] = struc_subset['session_number'].astype('int64')

## assign color per drug
color_selection = [color_map[index] for index in range(len(mapping))]

# %% CUSTOMIZE FOI
## Customized foi, add all foi's to dictionary
export_dir = r'C:\Users\Winnie\Desktop\QCB\Feature lists'
exp_df = pd.DataFrame(STRUC_FOI)
exp_df.to_csv(os.path.join(export_dir, '{}_STRUC_FOI.csv'.format(STRUCTURE)),
                 header=False, index = False)

"""
## sec61b
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
## golgi
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
## tubulin
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
## Dictionary of feature grouping
d = {'DNA Features': DNA_FOI, 
     'CELL Features': CELL_FOI,
     'DNA and CELL Features': DNA_FOI + CELL_FOI,
     'Structure Features': STRUC_FOI,
     'All Features': ALL_FOI,
     'Selected Structure Features': foi}

# %% Formatting df to run before PCA/ISOMAP/LDA

## Turn NaN values into 0's (with number of structure components = 0 and 1)
struc_subset_filled = struc_subset[ALL_FOI]
struc_subset_filled = struc_subset_filled.fillna(0, inplace=False)
#struc_subset_filled = struc_subset

## Add drug_label column - LDA will ignore this column
struc_subset_filled['drug_label'] = nom_cols['drug_label']
"""
## getting df with only vehicle and one other drug
only_drug = 'Brefeldin'

struc_subset_filled = struc_subset_filled[(struc_subset_filled.drug_label == only_drug) |
                        (struc_subset_filled.drug_label == 'Vehicle')]
struc_subset_filled = struc_subset_filled[(struc_subset_filled.drug_label == 'Vehicle')]
## dropping groups
ssf = struc_subset_filled
struc_subset_filled = ssf.drop(ssf[(ssf.drug_label == 'Brefeldin') |
                            (ssf.drug_label == 's-Nitro-Blebbistatin')].index,
                              inplace=False)
"""
# %% PCA
## Graphing PCA by above feature categories and getting tables
## run fillna block first

PCA_results = {}
sort_by = 'Abs(C1)'
for key, foi in d.items():
    struc_subset_copy = struc_subset_filled[foi]
    struc_scaled = qp.scaleFeaturesDF(struc_subset_copy)                           ## Scale features for pre-processing for PCA
    pca = PCA(n_components = 3, svd_solver = "randomized")
    pca.fit(struc_scaled)
    T = pca.transform(struc_scaled)
    ## Graph PCA and get correlation table of original features to each PC
    T_lab = qp.PlotT(T, color_selection, graphtype='PCA', 
                  addtotitle = ': {}'.format(key))
    Corr_Table = qp.GetCorrTable(struc_scaled[foi], T_lab, sort_by = sort_by)
    
    pca_expl_var = pca.explained_variance_ratio_
    PCA_results.update({'{}'.format(key): {'T_lab': T_lab,
                                            'pca_expl_var': pca_expl_var,
                                            'Corr_Table': Corr_Table}})


# %% LDA - Linear Discriminant Analysis - run fillna block first

LDA_results = {}
sortby = 'Abs(C1)'

for key, foi in d.items():
    lda_df = struc_subset_filled[foi]
    lda = LDA(n_components = 3)
    #T = lda.fit_transform(lda_df, y=nom_cols['drug_label'])
    T = lda.fit_transform(lda_df, y=struc_subset_filled['drug_label'])
    T_lab = qp.PlotT(T, color_selection, graphtype='LDA',
                  addtotitle = f'{key}', 
                  drug_lab = struc_subset_filled['drug_label'])
    Corr_Table = qp.GetCorrTable(struc_subset_filled[foi], T_lab, sort_by = sortby)
    exp_var_ratio = lda.explained_variance_ratio_
    weights = lda.coef_
    LDA_results.update({'{}'.format(key): {'T_lab': T_lab,
                                            'exp_var_ratio': exp_var_ratio,
                                            'Weights': weights,
                                            'Corr_Table': Corr_Table}})
    
# %% Get foi from top LDA features; get top foi to do analysis and plot later
 
sort_by = 'Abs(C1)'
top = 10
#table = LDA_results['All Features']['Corr_Table']
table = LDA_results['All Features']['Corr_Table']
foi = table.sort_values(by = [sort_by], ascending = False).head(top)['Feature']

# %% STATISTICS: RETURNS PARAMETERS_TABLE WITH MEAN AND STD OF TOP FOI ACROSSS DRUG GROUPS

## data table with foi
stats_df = struc_subset
stats_df['drug_label'] = nom_cols['drug_label']

foi_stats = {}
for feature in foi:
    table = stats_df.groupby(['drug_label'])[feature].describe()                #.unstack().reset_index()
    foi_stats[feature] = table
    
## Compile list of feature parameter into 1 table
parameters = ['mean', 'std']
parameters_table = {}
indices = foi_stats[list(foi_stats.keys())[0]].index.values.tolist()            ## list of drug indices

export = False
export_dir = r'C:\Users\Winnie\Desktop\QCB\Feature lists'

for par in parameters:
    parameter_table = {}    
    parameter_table = pd.DataFrame(index = indices)
    
    for feature, stats in foi_stats.items():
        parameter_table[feature] = stats[par]
    
    parameter_table = parameter_table.transpose()
    parameter_table = parameter_table[plot_order]                               ## re-order columns

    parameters_table.update({f'{par}': parameter_table})
    
    if export:
        exp_df = pd.DataFrame(parameters_table[par])
        exp_df.to_csv(os.path.join(export_dir, 'par_table_{}.csv'.format(par)), 
                      index = indices)
        
# %% PLOTTING SCATTER CI PLOTS WITH P-VALUES 
        
scatter_df = struc_subset
scatter_df['drug_label'] = nom_cols['drug_label']


"""
plot_order = ['Vehicle',
              'Brefeldin',
              'Paclitaxol',
              'Staurosporine']
"""

## Get features to plot 
#foi = ['str_number_of_components']                                     

## Plotting different cell sizes of different drugs
foi = ['str_volume_sum']

## Plot pointplots/scatter plots of top foi
qp.PlotScatterCI(scatter_df, foi)
 
# %% PLOTTING BY SCENE NUMBER OR SESSION NUMBER

## one plot per foi against scene number
#foi = ['str_number_of_components', 'str_volume_mean', 'str_volume_total']
foi = ['str_volume_sum']
struc_subset['drug_label'] = nom_cols['drug_label']
#drugs = ['Vehicle', 'Brefeldin', 'Paclitaxol', 'Staurosporine']
#drugs = ['Brefeldin']

time_sep = 'session_number'
for drug in drugs:

    ## Make subset of brefeldin dataset
    scatter_df = struc_subset[(struc_subset.drug_label == drug)]
    
    plot_order = list(sorted(set(scatter_df[time_sep])))
    
    qp.PlotScatterCI(scatter_df, foi, groupby=time_sep, addtotitle=f'{drug}')
# %% Plot all Vehicle together then Bref by session


foi = ['str_volume_sum']
struc_subset['drug_label'] = nom_cols['drug_label']
#drugs = ['Vehicle', 'Brefeldin', 'Paclitaxol', 'Staurosporine']
drugs = ['Brefeldin']

time_sep = 'session_number'
dff = struc_subset[(struc_subset.drug_label == drug) | (struc_subset.drug_label == 'Vehicle')]

dff.loc[dff.drug_label == 'Vehicle', 'session_number'] = 'Vehicle'

for drug in drugs:

    ## Make subset of brefeldin dataset
    scatter_df = struc_subset[(struc_subset.drug_label == drug)]
    
    plot_order = ['Vehicle', 1, 2, 3, 4]
    
    qp.PlotScatterCI(dff, foi, groupby=time_sep, addtotitle=f'{drug}')
# %% 2D Scatter plots  

plot_foi = ['str_number_of_components', 'str_volume_mean', 'str_volume_total']
plot_doi = ['Vehicle']
"""
scatter_df.drop(scatter_df.loc[scatter_df['drug_label'] == 
                               's-Nitro-Blebbistatin'].index, inplace=True)
"""
qp.PlotScatter2D(scatter_df, foi, plot_order)

# %% 2D Scatter plot with linear regression 
    ## with Ordinary Least Squares and Orthogonal Distance Regression options

scatter_df = struc_subset
scatter_df['drug_label'] = nom_cols['drug_label']
drug = 'Vehicle'

#plot_foi = ['str_number_of_components', 'str_volume_mean', 'str_volume_total']
plot_foi = ['str_volume_total']
x_lab = ['dna_volume']

for axis in x_lab:
        qp.PlotScatter2D(scatter_df, plot_foi, drug, x_lab=axis, linreg = True, 
                      odrreg=True)

for session in list(scatter_df['session_number'].unique()):
    dff = scatter_df.groupby('session_number').get_group(session)
    for axis in x_lab:
        qp.PlotScatter2D(dff, plot_foi, drug, x_lab=axis, linreg = True, 
                      odrreg=True, addtotitle = f'Session {session}')
        
# %% 3D Scatter plots                                                           ## features to plot

## 3D scatter plots over DNA volume and cell volume
plot_foi = STRUC_FOI  
qp.PlotScatter3D(scatter_df, plot_foi)
#PlotScatter3D(struc_subset, ['structure_meridional_eccentricity'], y_lab='dna_meridional_eccentricity')

# %% TEST AREA

# %%
# Parallel Coordinates Start Here:
plt.figure()
parallel_coordinates(compare_drug_scaled, 'drug_label')
plt.show()