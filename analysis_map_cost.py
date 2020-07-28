import pandas as pd
import numpy as np
from copy import deepcopy

# internal imports
from plots_general import plot_general
from plots_gain import plot_gains_map, plot_gains_map_max
from etl_preprocessing import extract_process_data


# get best gains of gain-max
def best_gain_gain(gains: pd.DataFrame):
    df_best_gain = pd.DataFrame(
        gains.groupby(
            ['iou_threshold']
        )['gain_score'].max()
    ).reset_index()  # best # gain of gain-max
    return df_best_gain


# get best gains of mAP-max
def best_gain_map(gains: pd.DataFrame):
    # get list of iou-con-map combos with best scores of each iou-map combo
    best_map = pd.DataFrame(
        gains.groupby(
            ['iou_threshold']
        )['m_ap'].max()).reset_index()  # best gain of mAP-max
    best_map = list(best_map.set_index(['iou_threshold', 'm_ap']).index)
    # get gain score for each of above combos
    df1 = pd.DataFrame(
        gains.groupby(
            ['iou_threshold', 'conf_threshold']
        )['m_ap'].max()).reset_index()  # best mAP for ea "iou x conf"
    df2 = pd.DataFrame(
        df1.merge(
            gains[['iou_threshold', 'conf_threshold', 'm_ap', 'gain_score']],
            how='left',
            on=['iou_threshold', 'conf_threshold', 'm_ap']))  # append gain info
    df3 = df2.set_index(['iou_threshold', 'm_ap']).loc[best_map].reset_index()
    df_best_map = pd.DataFrame(df3.groupby(['iou_threshold', 'm_ap'])['gain_score'].max()).reset_index()
    return df_best_map


def best_gain_map_avg(gains: pd.DataFrame):
    best_map = pd.DataFrame(
        gains.groupby(
            ['iou_threshold']
        )['m_ap'].max()).reset_index()  # best gain of mAP-max
    best_map = list(best_map.set_index(['iou_threshold', 'm_ap']).index)
    df = gains.set_index(['iou_threshold', 'm_ap']).loc[best_map]
    df_best_map_avg = pd.DataFrame(df.groupby(['iou_threshold'])['gain_score'].mean()).reset_index()
    return df_best_map_avg


def analysis_gain_map(
        gains_all: pd.DataFrame,
        gain_tp_weed: float = 0.5,
        gain_fp_weed: float = -0.3
):
    # narrow all-gains table to the fixed gains specified
    gains = gains_all[(gains_all['gain_tp_weed'] == gain_tp_weed) & (gains_all['gain_fp_weed'] == gain_fp_weed)]

    # get best gain results for each gain-type x IoU threshold
    df_best_gain = best_gain_gain(gains=gains)  # get best results for gain-max condition
    df_best_map = best_gain_map(gains=gains)  # get best results for map-max condition
    df_best_map_avg = best_gain_map_avg(gains=gains)

    return df_best_gain, df_best_map, df_best_map_avg

# using fixed gains, compare gain-max vs. map-max gain-types
def plot_analyze_gains(
        gains_all: pd.DataFrame,
        gain_tp_weed: float = 0.5,
        gain_fp_weed: float = -0.3
):

    # perform mAP & gain/gain analysis
    df_best_gain, df_best_map, df_best_map_avg = analysis_gain_map(
        gains_all=gains_all,
        gain_tp_weed=gain_tp_weed,
        gain_fp_weed=gain_fp_weed
    )

    # get df of rows only with the best-mAP
    df = gains_all[gains_all['m_ap'] == gains_all['m_ap'].max()]
    df['gain_score'].describe()

    # plot - gain-max vs. map-max - which one gives you the better gain?
    plot_gains_map(
        df_best_gain=df_best_gain,
        df_best_map=df_best_map_avg,
        close_plots=False
    )

    # plot - distribution of gains for data scenarios yielding max mAP
    plot_gains_map_max(
        gains_all=gains_all,
        close_plots=False
    )

    return df_best_gain, df_best_map, df_best_map_avg


if __name__ == "__main__":

    gain_tp_weed = 0.5
    gain_fp_weed = -0.3

    # extract data & gains
    df_results, gains_all, gains_best = extract_process_data(file_dir='results_iou_conf')
    gains_all.to_pickle('gains_all.pkl')  # save

    # plots - general analyses
    plot_general(
        df_results=df_results,
        df_gains=gains_all,
        close_plots=False
    )

    # plots - gains & mAP analyses
    df_best_gain, df_best_map, df_best_map_avg = plot_analyze_gains(
        gains_all=gains_all,
        gain_tp_weed=gain_tp_weed,
        gain_fp_weed=gain_fp_weed
    )

    print('========= Done =========')
