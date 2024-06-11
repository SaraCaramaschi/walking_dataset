import math
import os

import matplotlib
import pandas as pd
import pylab
from scipy import stats
from matplotlib import pyplot as plt
from scipy.stats import *

from settings import running_settings
from settings.running_settings import plots_settings

pylab.rcParams.update(plots_settings['params'])

matplotlib.use('TkAgg')

def compute_statistics(name, gt, est, percstats):
    df_stats = pd.DataFrame()
    error = est - gt
    errorperc=[]
    for g,ground in enumerate(gt):
        errorperc.append(np.round(((est[g] - ground)/ground)*100, 2))

    if percstats:
        error = np.array(errorperc)

    print('#####################################################')
    print('Statistic for ', name)
    print('mean: ', np.mean(error))
    print('median: ', np.median(error))
    mse = np.square(error).mean()
    print('RMSE: ', math.sqrt(mse))
    print('SD: ', np.std(error))
    print('ABS median: ', np.median(abs(error)))
    print('ABS mean: ', np.mean(abs(error)))
    print('ABS SD: ', np.std(abs(error)))
    print('ABS min: ', min(abs(error)))
    print('ABS max: ', max(abs(error)))
    print('ABS mean: ', np.mean(abs(error)))
    print('IQR range: ', stats.iqr(abs(error)))
    print('Q1(abs): ', np.percentile(abs(error), 25))
    print('Q3(abs): ', np.percentile(abs(error), 75))
    print('Q1: ', np.percentile((error), 25))
    print('Q3: ', np.percentile((error), 75))
    print('Correlation GT - EST: ', pearsonr(gt, est)[0])
    df_stats.at[0, 'mean'] = round(np.mean(error),2)
    df_stats.at[0, 'median'] = round(np.median(error),2)
    df_stats.at[0, 'RMSE'] = round(math.sqrt(mse),2)
    df_stats.at[0, 'SD'] = round(np.std(error),2)
    df_stats.at[0, 'ABS median'] = round(np.median(abs(error)),2)
    df_stats.at[0, 'ABS mean'] = round(np.mean(abs(error)),2)
    df_stats.at[0, 'ABS SD'] = round(np.std(abs(error)),2)
    df_stats.at[0, 'ABS min'] = round(min(abs(error)),2)
    df_stats.at[0, 'ABS max'] = round(max(abs(error)),2)
    df_stats.at[0, 'ABS mean'] = round(np.mean(abs(error)),2)
    df_stats.at[0, 'min'] = round(min(error), 2)
    df_stats.at[0, 'max'] = round(max(error), 2)
    df_stats.at[0, 'IQR(abs)'] = round(stats.iqr(abs(error)),2)
    df_stats.at[0, 'Q1(abs)'] = round(np.percentile(abs(error), 25),2)
    df_stats.at[0, 'Q3(abs)'] = round(np.percentile(abs(error), 75),2)
    df_stats.at[0, 'IQR'] = round(stats.iqr(error),2)
    df_stats.at[0, 'Q1'] = round(np.percentile((error), 25),2)
    df_stats.at[0, 'Q3'] = round(np.percentile((error), 75),2)
    df_stats.at[0, 'corr'] = round(pearsonr(gt, est)[0],2)
    df_stats.at[0, 'loa_low'] = np.mean(error) - 1.96 * np.std(error)
    df_stats.at[0, 'loa_high'] = np.mean(error) + 1.96 * np.std(error)
    df_stats.at[0, 'loa_low%'] = np.mean(errorperc) - 1.96 * np.std(errorperc)
    df_stats.at[0, 'loa_high%'] = np.mean(errorperc) + 1.96 * np.std(errorperc)
    df_stats.at[0, '0<er<50'] = len(np.where(abs(error) <= 50)[0])
    df_stats.at[0, '0<er<30'] = len(np.where(abs(error) <= 30)[0])
    df_stats.at[0, '30<er'] = len(np.where(abs(error) > 30)[0])
    df_stats.at[0, '50<er<100'] = len(np.where((abs(error) <= 100) & (abs(error) > 50))[0])
    df_stats.at[0, '100<er<200'] = len(np.where((abs(error) <= 200) & (abs(error) > 100))[0])
    df_stats.at[0, '200<er'] = len(np.where((abs(error) > 200))[0])
    df_stats = df_stats.transpose()
    return df_stats

def compute_stats(df):
    # Computing statistics of absolute values
    gt = df['distanceReference'].to_numpy()
    est = df['distanceByApp'].to_numpy()
    df_stats = compute_statistics('QSS-alltracks', gt, est, False)
    df_stats.columns = ['QSS-alltracks']

    # Computing statistics for each path
    for path in df['path curvature'].unique():
        df_path = df[df['path curvature'] == path]
        gt = df_path['distanceReference'].to_numpy()
        est = df_path['distanceByApp'].to_numpy()
        df_stats_path = compute_statistics(f'QSS-path:{path}', gt, est, False)
        df_stats_path.columns = [f'QSS-path:{path}']
        df_stats = pd.concat([df_stats, df_stats_path], axis=1)

    gt = df['distanceReference'].to_numpy()
    est = df['distanceByApp'].to_numpy()
    df_stats_perc = compute_statistics('QSS-alltracks-perc', gt, est, True)
    df_stats_perc.columns = ['QSS-alltracks-perc']

    # Computing statistics for each path in percentage
    for path in df['path curvature'].unique():
        df_path = df[df['path curvature'] == path]
        gt = df_path['distanceReference'].to_numpy()
        est = df_path['distanceByApp'].to_numpy()
        df_stats_path_perc = compute_statistics(f'QSS-path:{path}-perc', gt, est, True)
        df_stats_path_perc.columns = [f'QSS-path:{path}-perc']
        df_stats_perc = pd.concat([df_stats_perc, df_stats_path_perc], axis=1)

    df_stats = pd.concat([df_stats, df_stats_perc], axis=1)
    return df_stats


def error_path_analysis(info_all_tests):
    path_0_tests = info_all_tests[info_all_tests['path curvature'] == 0]
    path_0_tests = path_0_tests[path_0_tests['hasGNSS'] == True]
    path_1_tests = info_all_tests[info_all_tests['path curvature'] == 1]
    path_2_tests = info_all_tests[info_all_tests['path curvature'] == 2]

    err_0 = np.abs(path_0_tests['distanceByApp'] - path_0_tests['distanceReference'])
    err_1 = np.abs(path_1_tests['distanceByApp'] - path_1_tests['distanceReference'])
    err_2 = np.abs(path_2_tests['distanceByApp'] - path_2_tests['distanceReference'])
    err_0_perc = (np.abs(path_0_tests['distanceByApp'] - path_0_tests['distanceReference']) / path_0_tests[
        'distanceReference']) * 100
    err_1_perc = (np.abs(path_1_tests['distanceByApp'] - path_1_tests['distanceReference']) / path_1_tests[
        'distanceReference']) * 100
    err_2_perc = (np.abs(path_2_tests['distanceByApp'] - path_2_tests['distanceReference']) / path_2_tests[
        'distanceReference']) * 100

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    c = running_settings.plots_settings['color']['pastel_blue']
    # boxplots of the column distanceByApp for each path
    ax.boxplot([err_0, err_1, err_2], patch_artist=True, boxprops=dict(facecolor=c, color=c),
               capprops=dict(color=c),
               whiskerprops=dict(color=c),
               flierprops=dict(color='black', markeredgecolor='black'),
               medianprops=dict(color='black'))
    ax.set_xticklabels(['Straight - ' + str(path_0_tests.shape[0]),
                        'Gently curved - ' + str(path_1_tests.shape[0]),
                        'Curved - ' + str(path_2_tests.shape[0])])
    ax.set_ylabel('Absolute error QSS [m]', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_title('Absolute error QSS for each path type',
                 fontdict={'fontsize': running_settings.plots_settings['title_fontsize']})
    plt.show(block=False)
    plt.tight_layout()
    plt.savefig(running_settings.main_path + os.sep + 'images' + os.sep + 'error_path_boxplots.eps', format='eps',
                dpi=1000)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    c = running_settings.plots_settings['color']['pastel_blue']
    # boxplots of the column distanceByApp for each path
    ax.boxplot([err_0_perc, err_1_perc, err_2_perc], boxprops=dict(facecolor=c, color=c),
               patch_artist=True,
               capprops=dict(color=c),
               whiskerprops=dict(color=c),
               flierprops=dict(color='black', markeredgecolor='black'),
               medianprops=dict(color='black'))
    ax.set_xticklabels(['Straight - ' + str(path_0_tests.shape[0]),
                        'Gently curved - ' + str(path_1_tests.shape[0]),
                        'Curved - ' + str(path_2_tests.shape[0])])
    ax.set_ylabel('%', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_title('Absolute % error QSS for each path type',
                 fontdict={'fontsize': running_settings.plots_settings['title_fontsize']})
    plt.show(block=False)
    plt.tight_layout()
    plt.savefig(running_settings.main_path + os.sep + 'images' + os.sep + 'error_path_perc_boxplots.eps', format='eps',
                dpi=1000)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    # Scatter plot between distanceByApp and distanceReference with hue the column path
    import seaborn as sns
    sns.scatterplot(x='distanceReference', y='distanceByApp', hue='path curvature', data=info_all_tests, ax=ax,
                    palette=running_settings.plots_settings['palette'])
    ax.plot([0, 1000], [0, 1000], '--', color='black', label=f'$y = x$')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Straight', 'Gently curved', 'Curved', f'$y = x$'])
    ax.set_ylabel('Distance QSS [m]', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_xlabel('Reference Distance [m]', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_title('Distance QSS vs Reference Distance',
                 fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    plt.tight_layout()
    plt.savefig(running_settings.main_path + os.sep + 'images' + os.sep + 'analysis_error_path.eps', dpi=1000,
                format='eps')

    pass


def groundtruth_plotting(info_all_tests):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    ax[0].hist(info_all_tests['distanceReference'],
               bins=20,
               color=running_settings.plots_settings['color']['pastel_blue'], rwidth=0.95)
    ax[0].set_title('Ground truth distance', fontdict={'fontsize': running_settings.plots_settings['title_fontsize']})
    ax[0].set_ylabel('Number of walks', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax[0].set_xlabel('Distance [m]', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})

    ax[1].hist(info_all_tests['duration [s]'], bins=20, color=running_settings.plots_settings['color']['pastel_blue'],
               rwidth=0.95)
    ax[1].set_title('Tracks duration [s]', fontdict={'fontsize': running_settings.plots_settings['title_fontsize']})
    ax[1].set_ylabel('Number of walks', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax[1].set_xlabel('Duration [s]', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})

    ax[2].hist(info_all_tests['totSteps'], bins=20, color=running_settings.plots_settings['color']['pastel_blue'],
               rwidth=0.95)
    ax[2].set_title('Total number of steps', fontdict={'fontsize': running_settings.plots_settings['title_fontsize']})
    ax[2].set_ylabel('Number of walks', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax[2].set_xlabel('Number of steps', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})

    plt.show(block=False)
    plt.tight_layout()
    # saving plot
    plt.savefig(running_settings.main_path + os.sep + 'images' + os.sep + 'gt_dist_hist.eps',
                format='eps', dpi=1000)
    pass


def imu_gnss_total_gaps(info_all_tests):
    fig = plt.figure(figsize=(6, 5))
    import seaborn as sns
    ax = fig.add_subplot(111)
    sns.scatterplot(x='total_gaps_time_inertial', y='abs_perc_error', data=info_all_tests, ax=ax, color=running_settings.plots_settings['color']['pastel_blue'])
    gaps_inertial = info_all_tests[info_all_tests['total_gaps_time_inertial'].notna()]['total_gaps_time_inertial'].to_numpy()
    abs_error = info_all_tests[info_all_tests['total_gaps_time_inertial'].notna()]['abs_perc_error'].to_numpy()
    m, b = np.polyfit(gaps_inertial, abs_error, 1)
    ax.axline(xy1=(0, b), slope=m, color='black', linestyle='--') #  label=f'$y = {m:.1f}x {b:+.1f}$',
    print('Correlation between total gaps time inertial and abs perc error: ',
          np.corrcoef(gaps_inertial, abs_error))
    ax.set_xlabel('Total gaps time inertial [s]',
                  fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_ylabel('Absolute % error QSS', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_title('Total gaps time inertial vs Absolute % error QSS',
                 fontdict={'fontsize': running_settings.plots_settings['title_fontsize']})
    plt.show(block=False)
    plt.tight_layout()
    plt.savefig(running_settings.main_path + os.sep + 'images' + os.sep + 'IMUgapvsABSpercerror.eps', dpi=1000,
                format='eps')

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    sns.scatterplot(x='total_gaps_time_gnss', y='abs_perc_error', data=info_all_tests, ax=ax,
                    color=running_settings.plots_settings['color']['pastel_blue'])
    m, b = np.polyfit(info_all_tests[info_all_tests['total_gaps_time_gnss'].notna()]['total_gaps_time_gnss'],
                      info_all_tests[info_all_tests['total_gaps_time_gnss'].notna()]['abs_error'],
                      1)
    ax.axline(xy1=(0, b), slope=m, color='black', linestyle='--') # label=f'$y = {m:.1f}x {b:+.1f}$',
    print('Correlation between total gaps time GNSS and abs perc error: ',
          np.corrcoef(info_all_tests['total_gaps_time_gnss'], info_all_tests['abs_perc_error']))
    ax.set_xlabel('Total gaps time GNSS [s]', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_ylabel('Absolute % error QSS', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_title('Total gaps time GNSS vs Absolute % error QSS',
                 fontdict={'fontsize': running_settings.plots_settings['title_fontsize']})
    plt.show(block=False)
    plt.tight_layout()
    plt.savefig(running_settings.main_path + os.sep + 'images' + os.sep + 'GNSSgapvsABSpercerror.eps', dpi=1000,
                format='eps')

    pass


def fs_error(info_all_tests):
    fig = plt.figure(figsize=(6, 5))
    import seaborn as sns
    ax = fig.add_subplot(111)
    sns.scatterplot(x='fs_acc', y='abs_perc_error', data=info_all_tests, ax=ax, color=running_settings.plots_settings['color']['pastel_blue'])
    m, b = np.polyfit(info_all_tests[info_all_tests['fs_acc'].notna()]['fs_acc'].to_numpy(),
    info_all_tests[info_all_tests['fs_acc'].notna()]['abs_perc_error'].to_numpy(), 1)
    ax.axline(xy1=(0, b), slope=m, label=f'$y = {m:.1f}x {b:+.1f}$', color='black', linestyle='--')
    print('Correlation between fs inertial and abs perc error: ',
          np.corrcoef(info_all_tests[info_all_tests['fs_acc'].notna()]['fs_acc'].to_numpy(),  info_all_tests[info_all_tests['fs_acc'].notna()]['abs_perc_error'].to_numpy()))
    ax.set_xlabel('Fs IMU [Hz]',
                  fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_ylabel('Absolute % error QSS', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_title('Fs vs Absolute % error QSS',
                 fontdict={'fontsize': running_settings.plots_settings['title_fontsize']})
    plt.show(block=False)
    plt.tight_layout()
    plt.savefig(running_settings.main_path + os.sep + 'images' + os.sep + 'fs_imu_vs_percerr.eps', dpi=1000,
                format='eps')

    fig = plt.figure(figsize=(6, 5))
    import seaborn as sns
    ax = fig.add_subplot(111)
    sns.scatterplot(x='fs_gnss', y='abs_perc_error', data=info_all_tests, ax=ax, color=running_settings.plots_settings['color']['pastel_blue'])
    m, b = np.polyfit(info_all_tests[info_all_tests['fs_gnss'].notna()]['fs_gnss'],info_all_tests[info_all_tests['fs_gnss'].notna()]['abs_perc_error'], 1)
    ax.axline(xy1=(0, b), slope=m, color='black', linestyle='--') # label=f'$y = {m:.1f}x {b:+.1f}$',
    print('Correlation between fs_gnss and abs perc error: ',
          np.corrcoef(info_all_tests[info_all_tests['fs_gnss'].notna()]['fs_gnss'].to_numpy(),  info_all_tests[info_all_tests['fs_gnss'].notna()]['abs_perc_error'].to_numpy()))
    ax.set_xlabel('Fs GNSS [Hz]',
                  fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_ylabel('Absolute % error QSS', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_title('Fs GNSS vs Absolute % error QSS',
                 fontdict={'fontsize': running_settings.plots_settings['title_fontsize']})
    plt.show(block=False)
    plt.tight_layout()
    plt.savefig(running_settings.main_path + os.sep + 'images' + os.sep + 'gnssFsVsperc_error.eps', dpi=1000,
                format='eps')

    pass


def per_phone_error(info_all_tests):
    # add cluster_device column in info_all_tests
    info_all_tests.loc[:, 'cluster_device'] = info_all_tests['device']
    for i in range(info_all_tests.shape[0]):
        if 'apple' in info_all_tests['device'].iloc[i].lower() :
            info_all_tests['cluster_device'].iloc[i] = 'Apple'
        elif 'samsung' in info_all_tests['device'].iloc[i].lower() :
            info_all_tests['cluster_device'].iloc[i] = 'Samsung'
        elif 'huawei' in info_all_tests['device'].iloc[i].lower():
            info_all_tests['cluster_device'].iloc[i] = 'Huawei'
        elif 'oneplus' in info_all_tests['device'].iloc[i].lower():
            info_all_tests['cluster_device'].iloc[i] = 'OnePlus'
        elif 'google' in info_all_tests['device'].iloc[i].lower():
            info_all_tests['cluster_device'].iloc[i] = 'Google'
        elif 'oppo' in info_all_tests['device'].iloc[i].lower():
            info_all_tests['cluster_device'].iloc[i] = 'Oppo'
        elif 'xiaomi' in info_all_tests['device'].iloc[i].lower():
            info_all_tests['cluster_device'].iloc[i] = 'Xiaomi'

    # multiple conditions extracted from info_all_test

    straight_info_test = info_all_tests[(info_all_tests['path curvature'] == 0)]
    fig = plt.figure(figsize=(20, 18))
    # boxplot with hue the device column
    ax1 = fig.add_subplot(131)
    straight_info_test.boxplot(column='abs_perc_error', by='cluster_device', ax=ax1, patch_artist=True)
    # add count of samples for each group in the boxplot

    ax1.set_xticklabels(['Apple \n' + str(straight_info_test[straight_info_test['cluster_device'] == 'Apple'].shape[0]),
                        'OnePlus \n' + str(straight_info_test[straight_info_test['cluster_device'] == 'OnePlus'].shape[0]),
                         'Oppo \n' + str(straight_info_test[straight_info_test['cluster_device'] == 'Oppo'].shape[0]),
                         'Samsung \n' + str(
                             straight_info_test[straight_info_test['cluster_device'] == 'Samsung'].shape[0]),
                        'Xiaomi \n' + str(straight_info_test[straight_info_test['cluster_device'] == 'Xiaomi'].shape[0])
                        ])
    ax1.set_title('Straight path tracks')

    gently_curved_info_test = info_all_tests[(info_all_tests['path curvature'] == 1)]
    ax2 = fig.add_subplot(132)
    gently_curved_info_test.boxplot(column='abs_perc_error', by='cluster_device', ax=ax2, patch_artist=True)
    ax2.set_xticklabels(['Apple \n' + str(gently_curved_info_test[gently_curved_info_test['cluster_device'] == 'Apple'].shape[0]),
                         'Google \n' + str(
                             gently_curved_info_test[gently_curved_info_test['cluster_device'] == 'Google'].shape[0]),
                         'Huawei \n' + str(
                             gently_curved_info_test[gently_curved_info_test['cluster_device'] == 'Huawei'].shape[0]),
                         'OnePlus \n' + str(
                             gently_curved_info_test[gently_curved_info_test['cluster_device'] == 'OnePlus'].shape[0]),
                         'Oppo \n' + str(
                             gently_curved_info_test[gently_curved_info_test['cluster_device'] == 'Oppo'].shape[0]),

                         'Samsung \n' + str(gently_curved_info_test[gently_curved_info_test['cluster_device'] == 'Samsung'].shape[0]),
                        'Xiaomi \n' + str(gently_curved_info_test[gently_curved_info_test['cluster_device'] == 'Xiaomi'].shape[0])
                        ])
    ax2.set_title('Gently curved path tracks')

    curved_info_test = info_all_tests[(info_all_tests['path curvature'] == 2)]
    curved_apple = curved_info_test[curved_info_test['cluster_device'] == 'Apple']
    curved_apple.abs_perc_error.std()
    ax3 = fig.add_subplot(133)
    ax3.set_title('Curved path tracks')
    curved_info_test.boxplot(column='abs_perc_error', by='cluster_device', ax=ax3, patch_artist=True)
    ax3.set_xticklabels([
        'Apple \n' + str(curved_info_test[curved_info_test['cluster_device'] == 'Apple'].shape[0]),
        'Google \n' + str(curved_info_test[curved_info_test['cluster_device'] == 'Google'].shape[0]),
        'Huawei \n' + str(curved_info_test[curved_info_test['cluster_device'] == 'Huawei'].shape[0]),
    'OnePlus \n' + str(curved_info_test[curved_info_test['cluster_device'] == 'OnePlus'].shape[0]),
        'Oppo \n' + str(curved_info_test[curved_info_test['cluster_device'] == 'Oppo'].shape[0]),
                        'Samsung \n' + str(curved_info_test[curved_info_test['cluster_device'] == 'Samsung'].shape[0]),
                        'Xiaomi \n' + str(curved_info_test[curved_info_test['cluster_device'] == 'Xiaomi'].shape[0])
                        ])
    ax3.set_title('Curved path tracks')
    # share y axis between subplots
    ax1.set_xlabel('')
    fig.suptitle('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    ax1.set_ylabel('Absolute % error QSS', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax1.sharey(ax2)
    ax2.sharey(ax1)
    ax3.sharey(ax1)
    plt.tight_layout()
    plt.savefig(running_settings.main_path + os.sep + 'images' + os.sep + 'per_phone_error.eps', dpi=1000, format='eps')
    pass


def calculates_correlations(info_all_tests):
    corrs = pd.DataFrame()
    corrs.at[0, 'path_curv_corr'] = \
    stats.pearsonr(info_all_tests['path curvature'].to_numpy(), info_all_tests['abs_perc_error'].to_numpy())[0]
    corrs.at[0, 'time_gap_inertial_corr'] = stats.pearsonr(
        info_all_tests[info_all_tests['total_gaps_time_inertial'].notna()]['total_gaps_time_inertial'].to_numpy(),
        info_all_tests[info_all_tests['total_gaps_time_inertial'].notna()]['abs_perc_error'].to_numpy())[0]
    corrs.at[0, 'time_gap_gnss_corr'] = \
    stats.pearsonr(info_all_tests[info_all_tests['total_gaps_time_gnss'].notna()]['total_gaps_time_gnss'].to_numpy(),
                   info_all_tests[info_all_tests['total_gaps_time_gnss'].notna()]['abs_perc_error'].to_numpy())[0]
    corrs.at[0, 'fs_acc_corr'] = stats.pearsonr(info_all_tests[info_all_tests['fs_acc'].notna()]['fs_acc'].to_numpy(),
                                                info_all_tests[info_all_tests['fs_acc'].notna()][
                                                    'abs_perc_error'].to_numpy())[0]
    corrs.at[0, 'fs_gnss_corr'] = \
    stats.pearsonr(info_all_tests[info_all_tests['fs_gnss'].notna()]['fs_gnss'].to_numpy(),
                   info_all_tests[info_all_tests['fs_gnss'].notna()]['abs_perc_error'].to_numpy())[0]
    corrs.at[1, 'path_curv_corr'] = [True if p < 0.05 else False for p in [
        stats.pearsonr(info_all_tests['path curvature'].to_numpy(), info_all_tests['abs_perc_error'].to_numpy())[1]]][0]
    corrs.at[1, 'time_gap_inertial_corr'] = [True if p < 0.05 else False for p in [stats.pearsonr(
        info_all_tests[info_all_tests['total_gaps_time_inertial'].notna()]['total_gaps_time_inertial'].to_numpy(),
        info_all_tests[info_all_tests['total_gaps_time_inertial'].notna()]['abs_perc_error'].to_numpy())[1]]][0]
    corrs.at[1, 'time_gap_gnss_corr'] = [True if p < 0.05 else False for p in [stats.pearsonr(
        info_all_tests[info_all_tests['total_gaps_time_gnss'].notna()]['total_gaps_time_gnss'].to_numpy(),
        info_all_tests[info_all_tests['total_gaps_time_gnss'].notna()]['abs_perc_error'].to_numpy())[1]]][0]
    corrs.at[1, 'fs_acc_corr'] = [True if p < 0.05 else False for p in
                                  [stats.pearsonr(info_all_tests[info_all_tests['fs_acc'].notna()]['fs_acc'].to_numpy(),
                                                  info_all_tests[info_all_tests['fs_acc'].notna()]['abs_perc_error'].to_numpy())[1]]][0]
    corrs.at[1, 'fs_gnss_corr'] = [True if p < 0.05 else False for p in [
        stats.pearsonr(info_all_tests[info_all_tests['fs_gnss'].notna()]['fs_gnss'].to_numpy(),
                       info_all_tests[info_all_tests['fs_gnss'].notna()]['abs_perc_error'].to_numpy())[1]]][0]

    # replacing values
    info_all_tests['num_device'] = info_all_tests['cluster_device']
    info_all_tests['num_device'].replace(info_all_tests.num_device.unique(), range(len(info_all_tests.cluster_device.unique())), inplace=True)
    corrs.at[0, 'clust_device_corr'] = stats.pearsonr(info_all_tests['num_device'].to_numpy(),
                                                        info_all_tests['abs_perc_error'].to_numpy())[0]
    corrs.at[1, 'clust_device_corr'] = [True if p < 0.05 else False for p in [
        stats.pearsonr(info_all_tests['num_device'].to_numpy(), info_all_tests['abs_perc_error'].to_numpy())[1]]][0]

    corrs.to_csv(running_settings.main_path + os.sep + 'data' + os.sep + 'correlations.csv')

    pass


def speed_investigation(info_all_tests):
    fig = plt.figure(figsize=(6, 5))
    import seaborn as sns
    ax = fig.add_subplot(111)
    # sns.scatterplot(x='total_gaps_time_inertial', y='abs_perc_error', hue='path', data=info_all_tests, ax=ax, palette=running_settings.plots_settings['palette'])
    sns.scatterplot(x='average_walking_speed', y='abs_perc_error', data=info_all_tests, ax=ax, color=running_settings.plots_settings['color']['pastel_blue'])
    # Adding regression line
    # turn values into float 64 of info_all_tests[info_all_tests['total_gaps_time_inertial'].notna()]['total_gaps_time_inertial']
    # and info_all_tests[info_all_tests['total_gaps_time_inertial'].notna()]['abs_error']
    gaps_inertial = info_all_tests[info_all_tests['average_walking_speed'].notna()]['average_walking_speed'].to_numpy()
    abs_error = info_all_tests[info_all_tests['average_walking_speed'].notna()]['abs_perc_error'].to_numpy()
    # abs error to float
    m, b = np.polyfit(gaps_inertial, abs_error, 1)
    ax.axline(xy1=(0, b), slope=m, color='black', linestyle='--') #  label=f'$y = {m:.1f}x {b:+.1f}$',
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, ['Straight', 'Gently curved', 'Curved', f'$y = {m:.1f}x {b:+.1f}$'], loc='lower right')
    # print correlation values between total gaps time inertial and abs perc error
    print('Correlation between average_walking_speed and abs perc error: ',
          np.corrcoef(gaps_inertial, abs_error))
    ax.set_xlabel('Average_walking_speed [m/s]',
                  fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_ylabel('Absolute % error QSS', fontdict={'fontsize': running_settings.plots_settings['label_fontsize']})
    ax.set_title('Average_walking_speed vs Absolute % error QSS',
                 fontdict={'fontsize': running_settings.plots_settings['title_fontsize']})
    plt.show(block=False)
    plt.tight_layout()
    plt.savefig(running_settings.main_path + os.sep + 'images' + os.sep + 'averagespeedVSabspercerror.eps', dpi=1000,
                format='eps')
    pass


def analysis_QSS_algorithm(info_all_tests):

    info_all_tests.loc[:, 'error'] = info_all_tests['distanceReference'] - info_all_tests['distanceByApp']
    info_all_tests.loc[:, 'abs_error'] = np.abs(np.abs(info_all_tests['error']))
    info_all_tests.loc[:, 'index'] = info_all_tests.index
    info_all_tests.loc[:, 'abs_perc_error'] = (abs(info_all_tests['error']) / info_all_tests['distanceReference'])*100

    # Computing statistics
    df_stats = compute_stats(df=info_all_tests[info_all_tests['hasGNSS'] == True])
    # Save df_stats to csv
    df_stats.to_csv(running_settings.main_path + os.sep + 'data' + os.sep + 'df_stats_QSS1.csv')

    # Groundtruth plotting
    if True:
        groundtruth_plotting(info_all_tests)
    info_all_tests = info_all_tests[info_all_tests['hasGNSS'] == True]

    if True:
        per_phone_error(info_all_tests)
    # Error boxplots plotting
    if True:
        error_path_analysis(info_all_tests)

    # Scatter total_gaps_time_gnss vs abs_error
    if True:
        imu_gnss_total_gaps(info_all_tests)

    if True:
        fs_error(info_all_tests)

    if True:
        speed_investigation(info_all_tests)
    # correlations table:
    if True:
        calculates_correlations(info_all_tests)

    return None

