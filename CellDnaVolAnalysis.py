# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:28:31 2018

Analysis on cell size and dna volume with all drugs and structures

@author: winniel
"""

# %% Cell volume and dna volume analysis from different drugs
    
## first step: make sure you can aggregate the different structures
## plot cell_volume, DNA_volume, scaled cell_volume/dnavolume against drug

import_dir = r'\\allen\aics\microscopy\Winnie\QCB\Python Scripts\Drug datasets export'

df = pd.read_csv(os.path.join(import_dir, 'old_df.csv'), header = 0)


## check you have all structures - get count
per_struct_drug = {}
for struct_name, group in df.groupby('structure_name'):
    drug_dict = {}
    for drug, subgroup in group.groupby('drug_label'):
        drug_dict[drug] = subgroup
    per_struct_drug[struct_name] = drug_dict
    
## Count how many cells per condition    
counts_table = GetConditionCounts('structure_name')

dff = df
dff = dff.fillna(0, inplace=False)
dff['cell_to_dna_volume'] = dff['cell_volume']/dff['dna_volume']
#dff['drug_label'] = dff['drug_label'].replace({'S-Nitro-Blebbistatin': 's-Nitro-Blebbistatin'})
features = ['dna_volume', 'cell_volume', 'cell_to_dna_volume']
groupby = 'structure_name'

plot_order = list(dff['structure_name'].unique())

PlotScatterCI(dff, features, plot_order, groupby=groupby, ttest=False,
                  addtotitle=None)

## Need to remove Sec61b's outlier
sec61b = dff.groupby('structure_name').get_group('sec61b')
outlier_val = sec61b['dna_volume'].max()
dff.drop(dff[(dff.dna_volume == outlier_val)].index, inplace=True)

PlotScatterCI(dff, features, plot_order, groupby=groupby, ttest=False,
                  addtotitle=None)

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

PlotScatterCI(DMSO_subset, features, plot_order, groupby=groupby, ttest=False,
                  addtotitle=None)

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

# %% Extract each group by drug_label and plot confidence CI
    
os.chdir(r'\\allen\aics\microscopy\Winnie\QCB\Python Scripts')
savedir = r'C:\Users\winniel\Desktop\graphs'

#import qcbplotting as qp
    
drugs = dff['drug_label'].unique()
features = ['dna_volume', 'cell_volume', 'cell_to_dna_volume']

for drug in drugs:
    drug_subset = dff.groupby('drug_label').get_group(drug)

    qp.PlotScatterCI(drug_subset, features, groupby='structure_name', 
                  addtotitle=f'{drug}', plotallstruc=True, plot_order=None,
                  savegraphs=True, savedir=savedir)
    

