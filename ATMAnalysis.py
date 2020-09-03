# IRI ATM data analysis

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib.ticker import FormatStrFormatter
import os
from sklearn.utils import shuffle
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import lag_plot
from datetime import datetime
# from sktime.utils.plotting.forecasting import plot_ys
# from sktime.forecasting.model_selection import temporal_train_test_split


model = 'RF'
target = 'CLS1 ATM1'
register_matplotlib_converters()

# Implemented Functions
# => PLEASE DO NOT change anything in here.
# ----------------------------------------------------------------------------------------------------------------------

def add_new_features(df, target):
    wnd_SD_HD = []
    n_HDs_ahead = []
    for row in range(len(df)):
        wnd_SD_HD.append('No')
        n_HDs_ahead.append(0)
        # ----------------------
        i, tomorrow = 0, df.loc[row, 'Tomorrow SD HD Wnd']
        while tomorrow == 'Yes':
            n_HDs_ahead[row] += 1
            i += 1
            if row+i == len(df):
                tomorrow = 'No'
            else:
                tomorrow = df.loc[row+i, 'Tomorrow SD HD Wnd']
        if (row == len(df)-2) or (row == len(df)-1):
            n_HDs_ahead[row] += 3
        # ----------------------
        today = df.loc[row, 'Weekday']
        wnd1 = row + 5 - today
        wnd2 = row + 6 - today
        if today == 7:
            wnd1, wnd2 = row + 5, row + 6
        if (wnd1 > len(df)-1) or (wnd2 > len(df)-1):
            wnd_SD_HD[row] = 'Yes'
            continue
        if (df.loc[wnd1, 'Special day'] == 'Yes') or (df.loc[wnd1, 'Holiday'] == 'Yes'):
            wnd_SD_HD[row] = 'Yes'
        elif (df.loc[wnd2, 'Special day'] == 'Yes') or (df.loc[wnd2, 'Holiday'] == 'Yes'):
            wnd_SD_HD[row] = 'Yes'
        # ----------------------
    df['n HDs ahead'] = n_HDs_ahead
    df['Weekend SD HD'] = wnd_SD_HD
    df['1dayAgo'] = df.loc[:, target].shift()
    df['weekAgo'] = df.loc[:, target].shift(7)
    df['1dayAgo_diff'] = df.loc[:, '1dayAgo'].diff()
    return df
def clean_data(df, target):
    df = add_new_features(df, target)
    df['Date'] = pd.to_datetime(df['Date'])
    # df = df.astype({'Season': 'object', 'Month': 'object', 'Weekday': 'object'})
    # df = df.set_index('Date')
    # ---------------------------------
    ATMs = ['CLS1 ATM1', 'CLS1 ATM2', 'CLS2 ATM1', 'CLS3 ATM1', 'CLS4 ATM1', 'CLS4 ATM2', 'ATM (mean)']
    scales = []
    for atm in ATMs:
        scale = [df[atm].min(), df[atm].max()]
        temp = (df[atm] - scale[0]) / (scale[1] - scale[0])
        df = df.drop([atm], axis=1)
        df[atm] = temp
        scales.append(scale)
    # ---------------------------------
    df = df.reset_index(drop=True)
    return df, scales
def read_data(file_name, target):
    sheet_names = pd.ExcelFile('{}.xlsx'.format(file_name)).sheet_names
    df = pd.DataFrame()
    for sheet_name in sheet_names:
        if sheet_name == sheet_names[0]:
            df = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
        else:
            df2 = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
            df = pd.concat([df, df2], ignore_index=True, sort=False)
    df, scales = clean_data(df, target)
    # ---------------------------------
    root = 'summaryOfData'
    if not os.path.exists(root):
        os.makedirs(root)
    excel_output(df, root=root, file_name='dataInput')
    return df, scales
def features_stats(df):
    root = 'summaryOfData'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    statistics = pd.DataFrame()
    columns = ['Season', 'Month', 'Day of month', 'Day of year', 'Weekday', 'Special day', 'Holiday', 'Weekend',
               'Next 3days SD HD', 'Tomorrow SD HD Wnd', 'Weekend SD HD', 'n HDs ahead']
    for column in columns:
        if column == columns[0]:
            statistics = pd.DataFrame(df[column].value_counts()).reset_index(drop=False)
            statistics.rename(columns={'index': column, column: 'samples_num'}, inplace=True)
        else:
            temp = pd.DataFrame(df[column].value_counts()).reset_index(drop=False)
            temp.rename(columns={'index': column, column: 'samples_num'}, inplace=True)
            statistics = pd.concat([statistics, temp], axis=1)
    excel_output(statistics, root=root, file_name='statsFeatures')
    return statistics
def groupby_col(df, column):
    df1 = df.groupby([column]).mean().reset_index(drop=False)
    df2 = df.groupby([column]).count().reset_index(drop=False)
    return df1, df2
def view_data(df, target):
    root = 'summaryOfData'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    ATMs = ['ATM (2)', 'ATM (3)', 'ATM (4)', 'ATM (5)', 'ATM (6)', 'ATM (7)']
    for column in df.columns:
        if (column != 'PersianCal') and (column != target):
            fig, ax = plt.subplots(1, figsize=(12, 9))
            # ---------------------------------
            if column in ATMs:
                X = df[column]
                y = df[target]
                plt.scatter(X, y, c='gray', label=column)
                plt.plot([X.min(), X.max()], [y.min(), y.max()], '-r', linewidth=2.0, label='y = x')
                plt.xlabel('Cash Demands (IRI Currency) at {}'.format(column), fontsize=18)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            elif column == 'Date':
                X = df[column]
                y = df[target]
                plt.plot(X, y, 'ok-', linewidth=1.0, label=column)
                plt.xlabel(column, fontsize=18)
                plt.ylabel('Average Cash Demands (IRI Currency)', fontsize=18)
                ax.set_ylim(0, 1.1)
            else:
                df2, df3 = groupby_col(df, column=column)
                X = df2[column]
                y = df2[target]
                plt.plot(X, y, 'ok-', label='grouped by {}'.format(column))
                if column == 'n HDs ahead':
                    plt.xlabel('Number of holidays ahead', fontsize=18)
                else:
                    plt.xlabel(column, fontsize=18)
                ax.set_ylim(0, 1)
            # ---------------------------------
            plt.grid(linewidth=0.5)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylabel('Average Cash Demands (IRI Currency)', fontsize=18)
            plt.legend(loc='upper right', fontsize=14, fancybox=True, shadow=True)
            plt.title(label='Raw Data'.format(column), fontsize=21)
            plt.tight_layout()
            plt.savefig('{}/{}.png'.format(root, column))
            plt.close()
def excel_output(object, root, file_name):
    if root != '':
        object.to_excel('{}/{}.xls'.format(root, file_name))
    else:
        object.to_excel('{}.xls'.format(file_name))
# ----------------------------------------------------------------------------------------------------------------------
def select_features(df, target):
    aux = df[target]
    df = df.drop(['Date', 'ID', 'Next 3days SD HD', 'Tomorrow SD HD Wnd', 'Weekend SD HD',
                  'CLS1 ATM1', 'CLS1 ATM2', 'CLS2 ATM1', 'CLS3 ATM1', 'CLS4 ATM1', 'CLS4 ATM2', 'ATM (mean)'], axis=1)
    df[target] = aux
    df = df.dropna()
    return df
def encode_data(df, target):
    cat_index = df.columns[(df.dtypes == 'object').values]
    num_index = df.columns[(df.dtypes != 'object').values]
    if target in cat_index:
        cat_index = cat_index.delete(-1)
    elif target in num_index:
        num_index = num_index.delete(-1)
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    sc = MinMaxScaler()
    ct = make_column_transformer((ohe, cat_index), (sc, num_index), remainder='passthrough')
    # ct = make_column_transformer((ohe, cat_index), remainder='passthrough')
    ct.fit_transform(df)
    df2 = ct.transform(df)
    # ---------------------------------
    names = []
    for cat in cat_index:
        unique = df[cat].value_counts().sort_index()
        for name in unique.index:
            names.append('{}_{}'.format(cat, name))
    for num in num_index:
        names.append(num)
    names.append(target)
    # ---------------------------------
    df2 = pd.DataFrame(df2)
    df2.columns = names
    df2.index = df.index
    return df2
def split_data(df, time_series):
    if not time_series:
        df = shuffle(df)
    df_train = df[:730]
    df_test = df[730:]
    return df_train, df_test
def split_Xy(df, time_series):
    if not time_series:
        df = shuffle(df)
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1].to_numpy()
    return X, y
def compare_models(df, models, replica, cv, scoring, plotType, time_series):
    root = 'buildBestModel'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    results_acc = pd.DataFrame()
    for i in range(replica):
        print(i)
        X_train, y_train = split_Xy(df=df, time_series=time_series)
        results = []
        for name, model in models:
            cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            results.append(cv_results)
        if i == 0:
            results_acc = pd.DataFrame(results)
        else:
            results_acc = pd.concat([results_acc, pd.DataFrame(results)], axis=1, ignore_index=True)
    results = results_acc.values.tolist()
    results_acc['mean'] = results_acc.mean(axis=1)
    results_acc['std'] = results_acc.std(axis=1)
    # ---------------------------------
    best_score = 0
    best_model = [models[0][0], '', models[0][1]]
    j = 0
    legend = ''
    names = []
    for name, model in models:
        names.append(name)
        mean = results_acc['mean'][j]
        std = results_acc['std'][j]
        if j == 0:
            legend = '{}: {:.2f} +/- {:.2f}'.format(name, mean, std)
        else:
            legend += '\n{}: {:.2f} +/- {:.2f}'.format(name, mean, std)
        print('{}: {:.5f} +/- {:.5f}'.format(name, mean, std))
        if mean > best_score:
            best_score = mean
            best_model = [name, mean, std, model]
        j += 1
    # ---------------------------------
    if plotType == 'box':
        box_plot(results, names, legend, root, goal='compare_models')
    else:
        scatter_plot(results, best_model, names, cv, replica, root)
    return best_model
def grid_search(model):
    models = []
    hp1 = [10, 30, 50, 100]
    hp2 = ['auto', 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for n in hp1:
        for m in hp2:
            if model == 'RF':
                models.append(('{}_{}'.format(n, m), RandomForestRegressor(n_estimators=n, max_features=m)))
    return models
# ----------------------------------------------------------------------------------------------------------------------
def smooth(y_array, window):
    if window != 0:
        y_smoothed = pd.Series(y_array).rolling(window, center=False).mean().shift(-window).to_numpy()
    else:
        y_smoothed = y_array
    return y_smoothed
def box_plot(results, names, legend, root, goal):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    plt.boxplot(results, labels=names, sym='', medianprops=dict(color='lightgrey', linewidth=1.0),
                meanprops=dict(linestyle='-', color='red', linewidth=1.5), meanline=True, showmeans=True)
    if goal == 'compare_models':
        legend_ypos = 0.15
        plt.xticks(fontsize=18)
        plt.title('Algorithms Comparison', fontsize=21)
        figname = 'compareModels'
    else:
        legend_ypos = 0.25
        plt.grid(linewidth=0.5)
        plt.xticks(fontsize=12, rotation=45)
        plt.title('Sensitivity Analysis', fontsize=21)
        figname = 'sensitivity1'
    plt.text(0.95, legend_ypos, legend,
             ha='right', va='center', transform=ax.transAxes,
             fontdict={'color': 'k', 'size': 14},
             bbox={'boxstyle': 'round', 'fc': 'mistyrose', 'ec': 'red', 'pad': 0.5})
    ax.set_ylim(-0.5, 1)
    plt.yticks(fontsize=16)
    plt.ylabel('Accuracy (R2)', fontsize=18)
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(root, figname))
    plt.close()
def scatter_plot(results, best_model, names, cv, replica, root):
    results = pd.concat([pd.DataFrame(names), pd.DataFrame(results)], axis=1)
    results['mean'] = results.mean(axis=1)
    results['std'] = results.std(axis=1)
    results['comb'] = np.arange(1, len(results)+1)
    # ---------------------------------
    info = "R2 = {:.2f} +/- {:.2f} (best combination) \n{} fold cross validation \n{} replicas per combination" \
        .format(best_model[1], best_model[2], cv, replica)
    # ---------------------------------
    fig, ax = plt.subplots(1, figsize=(12, 9))
    X = results['comb']
    y = results['mean']
    plt.plot(X, y, '-o', c='black', linewidth=1.0, label='Cross validation set')
    plt.text(0.05, 0.91, info,
             ha='left', va='center', transform=ax.transAxes,
             fontdict={'color': 'k', 'size': 14},
             bbox={'boxstyle': 'round', 'fc': 'mistyrose', 'ec': 'red', 'pad': 0.5})
    # ---------------------------------
    plt.grid(linewidth=0.5)
    plt.xticks(fontsize=16)
    plt.yticks(np.linspace(0.8, 1, num=5), fontsize=14)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.set_ylim(0.80, 1.0)
    plt.xlabel('Combination', fontsize=18)
    plt.ylabel('Accuracy (R2)', fontsize=18)
    plt.legend(loc='upper right', fontsize=14, fancybox=True, shadow=True)
    plt.title('Hyper-parameters Tuning', fontsize=21)
    plt.savefig('{}/tuneHPs.png'.format(root))
    plt.close()
    excel_output(results, root=root, file_name='tuneHPs')
def feature_importance(df, estimator):
    root = 'buildBestModel'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    names = df.columns
    imp = estimator.feature_importances_
    indices = np.argsort(imp)
    # ---------------------------------
    fig, ax = plt.subplots(1, figsize=(12, 9))
    plt.barh(range(len(indices)), imp[indices], color='red', align='center')
    plt.xticks(fontsize=16)
    plt.yticks(range(len(indices)), [names[i] for i in indices], fontsize=12)
    plt.xlabel('Relative Importance', fontsize=21)
    plt.title('Features Importance', fontsize=25)
    plt.tight_layout()
    plt.savefig('{}/featureImportance.png'.format(root))
    plt.close()
    # ---------------------------------
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap='coolwarm', square=True)
    plt.title('Correlation matrix', fontsize=25)
    plt.tight_layout()
    plt.savefig('{}/correlationMatrix.png'.format(root))
    plt.close()
def parity_plot(y_test, y_pred):
    root = 'buildBestModel'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    errors = [('R2', r2_score(y_test, y_pred)),
              ('MSE', mean_squared_error(y_test, y_pred)),
              ('MAE', mean_absolute_error(y_test, y_pred)),
              ('RMSE', np.sqrt(mean_squared_error(y_test, y_pred)))]
    info = '{} = {:.2f} \n{} = {:.2f} \n{} = {:.2f} \n{} = {:.2f}'.format(errors[0][0], errors[0][1],
                                                                          errors[1][0], errors[1][1],
                                                                          errors[2][0], errors[2][1],
                                                                          errors[3][0], errors[3][1],)
    # ---------------------------------
    fig, ax = plt.subplots(1, figsize=(12, 9))
    # y_test = (scales[6][1]-scales[6][0]) * y_test + scales[6][0]
    # y_pred = (scales[6][1] - scales[6][0]) * y_pred + scales[6][0]
    plt.scatter(y_test, y_pred, c='gray', label='Testing set')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '-r', linewidth=2.0, label='y = x')
    plt.text(0.05, 0.9, info,
             ha='left', va='center', transform=ax.transAxes,
             fontdict={'color': 'k', 'size': 14},
             bbox={'boxstyle': 'round', 'fc': 'mistyrose', 'ec': 'red', 'pad': 0.5})
    # ---------------------------------
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # ax.set_xlim(10**7, 10**9)
    # ax.set_ylim(10**7, 10**9)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('True Label', fontsize=18)
    plt.ylabel('Prediction', fontsize=18)
    plt.legend(loc='upper right', fontsize=14, fancybox=True, shadow=True)
    plt.title('Cash Demand (IRI Currency)', fontsize=21)
    plt.savefig('{}/parityPlot.png'.format(root))
    plt.close()
    # ---------------------------------
    df = pd.DataFrame(columns=['Experiment', 'Prediction'])
    df['Experiment'] = y_test
    df['Prediction'] = y_pred
    excel_output(df, root=root, file_name='parityPlot')


# ----------------------------------------------------------------------------------------------------------------------

# reading data + analysis
dataInput, scales = read_data('dataATM', target)

# data summary
# features_stats(df=dataInput)
# view_data(df=dataInput, target=target)

# encoding and splitting data
cashDemand = select_features(dataInput, target)
cashDemand = encode_data(cashDemand, target)
training, testing = split_data(df=cashDemand, time_series=True)

# ----------------------------------------------------------------------------------------------------------------------

best_algorithm = RandomForestRegressor()
X_train, y_train = split_Xy(training, True)
best_algorithm.fit(X_train, y_train)
# X_test, y_test = split_Xy(testing, True)
# y_pred = best_algorithm.predict(X_test)
# feature_importance(training, best_algorithm)
# parity_plot(y_test, y_pred)
#
# fh = np.arange(len(y_test)) + 1
# y_test = pd.Series(y_test, index=pd.Series(y_train).index[-1]+fh)
# y_pred = pd.Series(y_pred, index=pd.Series(y_train).index[-1]+fh)
# plot_ys(pd.Series(y_train), y_test, labels=["y_train", "y_test"])
# plot_ys(pd.Series(y_train), y_test, y_pred, labels=["y_train", "y_test", "y_pred"])



# comparing different models
# models = [('ANN', MLPRegressor(max_iter=2000)),
#           ('KNN', KNeighborsRegressor()),
#           ('RF', RandomForestRegressor(max_features=0.2)),
#           ('SVM', SVR())]
# tscv = TimeSeriesSplit(n_splits=10)
# best_algorithm = compare_models(training, models, 2, 5, 'r2', 'box', True)

# tuning hyper-parameters and building the best model
# models = grid_search(model)
# tscv = TimeSeriesSplit(n_splits=10)
# best_estimator = compare_models(df=training, models=models, replica=10, cv=5, scoring='r2', plotType='scatter',
#                                 time_series=False)

# features importance
# X_train, y_train = split_Xy(training, True)
# best_algorithm[3].fit(X_train, y_train)
# feature_importance(training, best_algorithm[3])

# parity plot
last_week = []
for i in range(7):
    last_week.append(y_train[len(y_train)-7+i])
X_test = pd.DataFrame()
y_pred = []
for sample in range(len(testing)):
    print(sample)
    X_temp = testing.iloc[sample, :-1]
    X_temp = pd.DataFrame(X_temp).transpose()
    X_temp['1dayAgo'], X_temp['weekAgo'], X_temp['1dayAgo_diff'] = last_week[6], last_week[0], last_week[6]-last_week[5]
    y_temp = best_algorithm.predict(X_temp)
    X_test = pd.concat([X_test, X_temp], ignore_index=True)
    y_pred.append(y_temp[0])
    last_week.append(y_temp[0])
    del last_week[0]

y_pred = np.asarray(y_pred)
X_test_original, y_test = split_Xy(testing, True)
y_pred_original = best_algorithm.predict(X_test_original)
parity_plot(y_test, y_pred)
parity_plot(y_test, y_pred_original)

fh = np.arange(len(y_test)) + 1
y_test = pd.Series(y_test, index=pd.Series(y_train).index[-1]+fh)
y_pred = pd.Series(y_pred, index=pd.Series(y_train).index[-1]+fh)
y_pred_original = pd.Series(y_pred_original, index=pd.Series(y_train).index[-1]+fh)
plot_ys(pd.Series(y_train), y_test, labels=["y_train", "y_test"])
plot_ys(pd.Series(y_train), y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
plot_ys(pd.Series(y_train), y_test, y_pred_original, labels=["y_train", "y_test", "y_pred"])

# ----------------------------------------------------------------------------------------------------------------------

# sensitivity analysis
# features = [('Original', ''),
#             ('Expos. time', 'Exposure_time'),
#             ('pH', 'pH'),
#             ('Acetic acid', 'Total_acetic_acid'),
#             ('pH + A.acid', ['pH', 'Total_acetic_acid']),
#             ('Temperature', 'Temperature'),
#             ('CO2 p.pres.', 'CO2_partial_pressure'),
#             ('H2S p.pres.', 'H2S_partial_pressure'),
#             ('Salt (NaCl)', 'Salt_(NaCl)'),
#             ('Inhib conc.', 'Inhibitor_Concentration')]
# sensitivity1(df=cashDemand, test_size=0.2, estimator=best_estimator, features=features, replicas=20)

# ----------------------------------------------------------------------------------------------------------------------

# the end
print('=> DONE!')

# ----------------------------------------------------------------------------------------------------------------------
