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
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import scipy.odr
import seaborn as sns
import sklearn.preprocessing as preprocessing
from scipy import stats


def get_condition_counts(df, *features):
    """

    :param df:
    :param features:
    :return:
    """

    """
    Function groups df by list of input features and returns a dataframe
    with the count of each grouped subset.

    Example:
    >>> get_condition_counts(df, 'structure_name', 'drug_label')
    This gives number of cells for each drug_label per structure
    Parameters

    :param df: pd.DataFrame

    :param features: str

    :return: pd.dataframe with counts, grouped by the list of inputs

    """

    conditions = df.groupby(list(features)).size().reset_index(name='Count')
    return conditions


def get_corr_coeff(struc_subset, T):  # T is 1-D list of one of PC
    """
    Function calculates correlation coefficient between every column of struc_subset and a column vector of T.

    :param struc_subset: pd.dataframe

    :param T: 1-D list of one of the components from PCA or LDA

    :return: list of correlation coefficients
    """
    corr_coeff = []
    col_list = list(struc_subset)
    for col in col_list:
        corr = np.corrcoef(struc_subset[col], T)[1, 0]  # just corrcoeff returns a 2x2 matrix (diagonal 1's)
        corr_coeff.append(corr)
    return corr_coeff


def scale_features_df(df):
    """
    Function for pre-processing dataframe before PCA. It uses sklearn's preprocessing standard scaler to remove
    the mean and scale to unit variance for all the features (will have mean of 0 and standard deviation of 1).
    Returns scaled dataset.

    :param df: pd.dataframe
    :return: scaled pd.dataframe
    """
    scaled = preprocessing.StandardScaler().fit_transform(df)
    scaled = pd.DataFrame(scaled, columns=df.columns)

    return scaled


def plot_T(T, color_selection, structure, mapping, graphtype=None, addtotitle=None, drug_lab=None, savegraphs=False,
           savedir=Path('')):
    """
    Function will take  plot histogram, 2D, or 3D graph of dimensionality reduction results from PCA or LDA.
    Graph type will depend on dimensionality of transformed T (dimensionality is number of classes - 1).
    For example, histogram will be plotted if T is reduced to 1-D from having 2 classes (e.g. 2 drug groups).
    Function will tag strings to the graph title to distinguish between graphs if they're supplied.

    :param T: transformed array from sklearn's lda or pca fit_transform function. Each row represents 1 cell. Columns
    are that cell's value in component 1, 2, 3, etc.

    :param color_selection: list of strings denoting plotting color with length equals to number of drug_labels in that
    structure group

    :param structure: string variable of structure being analyzed. Must match format as appear in dataframe.

    :param mapping: List of drug_label strings

    :param graphtype: string to tag onto graph title. E.g. 'PCA', 'LDA', etc

    :param addtotitle: string to tag onto graph title telling what is being plotted. e.g. "DNA Features",
    "Cell Features", "Structure Features"

    :param drug_lab: series of drug_label names.

    :param savegraphs: boolean. Default False will not save graphs. True will save out graphs to png.

    :param savedir: Path object of directory to save out graphs to

    :return: pd.dataframe same length as the number of cells in the structure group. Each row of T_lab would have
    the drug group that the cell belongs too, as well as its values in C1, 2, 3, as applicable
    """

    dimensionality = T.shape[1]
    column_data = {f'C{i + 1}': T[:, i] for i in range(dimensionality)}
    T_lab = pd.DataFrame({'drug_label': drug_lab,
                          **column_data})  # ** for expands dictionary

    fig = plt.figure()
    projection = '3d' if dimensionality == 3 else None
    ax = fig.add_subplot(111, projection=projection)

    if addtotitle is None:
        title = f'{graphtype}__{structure}'

    else:
        title = f'{graphtype} of {structure}__{addtotitle}'

    ax.set_title(title)
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

    # Plot legend
    plt.legend()

    if savegraphs:
        fig.set_size_inches(10, 6)
        fig.savefig(savedir / f'{title}.png')
    else:
        # Plot
        plt.show()

    return T_lab


def get_corr_table(struc_subset, T_lab, sort_by='Abs(C1)'):
    """
    Function takes in struc_subset dataframe and reference T_lab dataframe to compute correlation coefficients
    between each transformed component to each feature column in the original, untransformed data.
    There is one correlation coefficient per feature column to each transformed component.

    :param struc_subset: pd.dataframe

    :param T_lab: pd.dataframe of transformed, dimension-reduced T labeled with drug group for each cell

    :param sort_by: string of column title to sort correlation table descending by

    :return: pd.dataframe of correlation table for each feature to each component.
    Sorted descending by column specified.
    """
    dimensionality = T_lab.shape[1] - 1  # subtracted one because first column is added drug_label

    c_table = {}
    c_table.update(
        {'C1_corr': get_corr_coeff(struc_subset, T_lab['C1'])})  # same correlation using original scaled or unscaled
    if dimensionality > 1:
        c_table.update({'C2_corr': get_corr_coeff(struc_subset, T_lab['C2'])})
    if dimensionality > 2:
        c_table.update({'C3_corr': get_corr_coeff(struc_subset, T_lab['C3'])})

    corr_data_abs = {f'Abs(C{i + 1})': [abs(x) for x in c_table[f'C{i + 1}_corr']] for i in range(dimensionality)}
    corr_data = {f'C{i + 1}': c_table[f'C{i + 1}_corr'] for i in range(dimensionality)}

    corr_table = pd.DataFrame({'Feature': list(struc_subset),
                               **corr_data_abs,
                               **corr_data})

    corr_table_sorted = corr_table.sort_values(by=[sort_by],
                                               ascending=False)
    return corr_table_sorted


def get_plot_order(df, control, by='drug_label'):
    """
    Function will take df dataframe, control group string name, and group_by parameter and return a list of
    strings denoting plot order with control being first element

    :param df: pd.dataframe

    :param control: string name of control group

    :param by: string denoting feature name to group by
    :return: list of strings denoting plot order with control being first element
    """
    plot_order = [control]
    for drug in list(df[by].unique()):
        if drug not in plot_order:
            plot_order.append(drug)

    return plot_order


def plot_scatter_ci(df, structure, features, control='Vehicle', groupby='drug_label',
                    addtotitle=None, plotallstruc=False, plot_order="",
                    savegraphs=False, savedir=Path(''), y_lim=None):
    """
    Plot scatter plots with seaborn module for each category with 95% confidence interval.

    :param df: pd.dataframe

    :param structure: string name of structure currently being analyzed

    :param features: list of features to plot each graph in

    :param control:string name of control group, if applicable

    :param groupby:string name as parameter/category to group df by

    :param addtotitle: string tag to the title

    :param plotallstruc: boolean. Default False for looking at features within one structure group.
    True would give option to plot all structures on the same graph

    :param plot_order: list of strings denoting plot_order of graphs

    :param savegraphs: boolean. Default False will not save graphs. True will save out graphs to png.

    :param savedir: Path object of directory to save out graphs to

    :param y_lim: tuple. parameter with ymin and ymax for normalizing scale on all graphs

    :return: None

    """
    if plot_order == "":
        plot_order = get_plot_order(df, control, by=groupby)

    # Set up the matplotlib figure
    sns.set(style="darkgrid")

    for feature in features:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # To make sure all plots are on same scale, hard-code y-limit
        if y_lim is not None:
            plt.ylim(*y_lim)

        if addtotitle is None:
            if plotallstruc:
                title = f'{feature}'
            else:
                title = f'{structure}__{feature}'

        else:
            if plotallstruc:
                title = f'{feature}__{addtotitle}'
            else:
                title = f'{structure}__{feature}__{addtotitle}'

        ax.set_title(title)
        ax = sns.pointplot(x=groupby, y=feature, data=df, ci=95, size=3,
                           color='k', join=False, order=plot_order,
                           estimator=np.mean)

        # Make sure pointplot on top
        plt.setp(ax.lines, zorder=100)
        plt.setp(ax.collections, zorder=100, label="")

        sns.stripplot(x=groupby, y=feature, data=df, jitter=True,
                      palette="Pastel1", edgecolor='k', size=5,
                      order=plot_order)

        if savegraphs:
            fig.set_size_inches(10, 6)
            fig.savefig(savedir / f'{title}.png')

        plt.show()


def plot_scatter_2d(df, structure, mapping, color_selection, plot_foi, *doi,
                    x_lab='dna_volume', odrreg=False, addtotitle=None,
                    savegraphs=False, savedir=Path(''), y_lim=None):
    """
    Function will plot 2D scatter plots with y-axis being each of the features provided and x-axis being dna_volume by
    default or user's input. If skipped plotting any drug group, will print message with error to console.

    :param df: pd.dataframe

    :param structure: string name of structure currently being analyzed

    :param mapping: List of drug_label strings

    :param color_selection: list of string denoting plotting color with length equals to number of drug_labels
    in that structure group

    :param plot_foi: list of feature of interest to plot. Generates one plot per foi

    :param doi: string extension of drug of interest to plot

    :param x_lab: string name to go on x-axis. Default "dna_volume"

    :param odrreg: boolean. Default False. True would plot an odrreg line with R^2 value and slope

    :param addtotitle: optional. string to tag to graph title

    :param savegraphs: boolean. Default False will not save graphs. True will save out graphs to png.

    :param savedir: Path object of directory to save out graphs to

    :param y_lim: tuple. parameter with ymin and ymax for normalizing scale on all graphs

    :return: None
    """
    if isinstance(doi[0], list):
        doi = doi[0]

    for foi in plot_foi:

        # To make sure all plots are on same scale, hard-code y-limit
        if y_lim is not None:
            plt.ylim(*y_lim)

        for drug in doi:
            index = mapping.index(drug)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            y_lab = foi

            if addtotitle is None:
                title = f'{structure}__{foi}__{drug}'
            else:
                title = f'{structure}__{foi}__{addtotitle}__{drug}'

            ax.set_title(title)
            ax.set_xlabel(f'{x_lab}')
            ax.set_ylabel(f'{y_lab}')

            try:

                drug_group = df.groupby('drug_label').get_group(drug)

                color = color_selection[index]

                ax.scatter(drug_group[x_lab], drug_group[y_lab],
                           c=color, label=drug)

                x = drug_group[x_lab]
                y = drug_group[y_lab]

                x_norm = x / np.mean(x)
                y_norm = y / np.mean(y)

                slope, intercept, r_value, p_value, std_err = stats.linregress(x_norm, y_norm)
                r_sq = r_value ** 2

                if odrreg:
                    # Model for fitting
                    linear_model = scipy.odr.Model(linear_f)

                    # Real Data Object
                    data = scipy.odr.Data(x, y)

                    # Set up ODR with model and data
                    odr = scipy.odr.ODR(data, linear_model, beta0=[0, 1])
                    odr.set_job(fit_type=0)
                    out = odr.run()

                    # Generate fitted data
                    y_fit = linear_f(out.beta, x)
                    # y_fit = linear_f(out.beta, x_norm)
                    # ax.plot(x, y_fit*np.mean(y), c='k', label='ODR')
                    odrslope = out.beta[0]
                    ax.plot(x, y_fit, c='k', label=f'ODR. $R^2$: {r_sq:.3f}, slope={odrslope:.3f}')

                plt.legend()

            except Exception as err:
                print(f'Skipped plotting {drug}: ({err})')
                pass

            if savegraphs:
                fig.set_size_inches(10, 6)
                fig.savefig(savedir / f'{title}.png')


def linear_f(p, x):
    """
    Function takes p and x to get slope, constant, and array of x-values to generate a linear function

    :param p: slope and constant

    :param x: series of x values

    :return: series of y values linearly mapped to input x values
    """
    m, c = p
    print(type(p))
    return m * x + c


def plot_scatter_3d(df, structure, mapping, color_selection, plot_foi,
                    x_lab='dna_volume', y_lab='cell_volume', savegraphs=False,
                    savedir=Path('')):
    """
    Function takes in dataframe df and list of features in plot_foi to plot 3D plots against dna_volume and cell_volume
    with the specified structure group and color_selection mapped to each drug_label from mapping list

    :param df: pd.dataframe

    :param structure: string name of structure currently being analyzed

    :param mapping: List of drug_label strings

    :param color_selection: list of string denoting plotting color with length equals to number of drug_labels in that
    structure group

    :param plot_foi: list of features to plot

    :param x_lab: string name of feature to plot on x-axis. Default value is 'dna_volume'

    :param y_lab: string name of feature to plot on y-axis. Default value is 'cell_volume'

    :param savegraphs: boolean. Default False will not save graphs. True will save out graphs to png.

    :param savedir: Path object of directory to save out graphs to

    :return: None
    """
    for foi in plot_foi:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z_lab = foi
        title = f'{structure}__{foi}'

        ax.set_title(title)
        ax.set_xlabel(f'{x_lab}')
        ax.set_ylabel(f'{y_lab}')
        ax.set_zlabel(f'{z_lab}')

        for drug in mapping:
            index = mapping.index(drug)
            try:
                drug_group = df.groupby('drug_label').get_group(drug)
                color = color_selection[index]
                ax.scatter(drug_group[x_lab], drug_group[y_lab],
                           drug_group[z_lab], c=color, label=drug)
            except Exception as err:
                print(f'Skipped plotting {drug}: ({err})')
                pass

        if savegraphs:
            fig.set_size_inches(10, 6)
            fig.savefig(savedir / f'{title}.png')

        # Plot
        plt.legend()
        plt.show()