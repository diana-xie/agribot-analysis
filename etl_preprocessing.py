import pandas as pd
import numpy as np
import re
import os
from copy import deepcopy
import logging

# setup
DATA_REGEX = {'iou_threshold': "mAP@(\d+\.\d+)",
              'conf_threshold': "for conf_thresh = (\d+\.\d+), precision",
              'ap_corn': "name = corn, ap = (\d+\.\d+)",  # average precision, corn
              'ap_weed': "name = weed, ap = (\d+\.\d+)",
              'tp_corn': "name = corn.+\(TP = (\d+)",  # true positives
              'tp_weed': "name = weed.+\(TP = (\d+)",
              'fp_corn': "name = corn.+FP = (\d+)",  # false positives
              'fp_weed': "name = weed.+FP = (\d+)",
              'average_iou': "average IoU = (\d+\.\d+)",
              'm_ap': "mean average precision.+ = (\d+\.\d+)",
              'precision': ", precision = (\d+\.\d+),",
              'recall': ", recall = (\d+\.\d+), ",
              'f1': "F1-score = (\d+\.\d+)"
              }


def extract_data(files: list):
    """
    extract the data from text files
    Parameters
    ----------
    files: list of files

    """
    # extract data
    results = []
    for filename in files:
        # open data file
        try:
            f = open('results_iou_conf/' + filename, "r")
            text = f.read()
        except Exception as ex:
            logging.error("Error reading file '{}': {}".format(filename, ex))
            raise ex
        # extract data using regex
        data_extracted = {}
        for variable_name, variable_regex in DATA_REGEX.items():
            try:
                data_extracted[variable_name] = re.compile(variable_regex).findall(text)[0]
            except Exception as ex:
                logging.error("Error getting data from file '{}': {}".format(filename, ex))
                data_extracted[variable_name] = 'nan'
        results.append(data_extracted)

    # postprocessing
    results = pd.DataFrame(results).astype(float)
    results = results[results['precision'].notnull()]

    return results


def convert_to_numeric(data: pd.DataFrame, columns=None):
    """
    Converts data from results, to numeric. Also performs scaling
    Parameters
    ----------
    data: dataframe of results, which were extracted from txt files in extract_data()
    columns: list of columns to convert to numeric and decimal (/100)

    """
    if columns is None:
        columns = ['ap_corn', 'ap_weed', 'average_iou']
    data[columns] = data[columns] / 100  # % to decimal
    return data


def compute_gain(data: pd.DataFrame, gain_tp_weed: float = 0.5, gain_fp_weed: float = -0.3):
    """
    compute "gain" score
    Parameters
    ----------
    data: df of results, extracted from txt files of inference
    gain_tp_weed: gain saved, when correctly killing 1 weed plant
    gain_fp_weed: gain incurred, when incorrectly killing 1 corn plant (thus should always be negative value)

    Returns
    -------
    df with new "gain_score" col.positive values indicate $ gained. negative means it's losing $
    """
    data['gain_score'] = (gain_tp_weed * data["tp_weed"]) + (gain_fp_weed * data["fp_weed"])
    data['gain_tp_weed'] = gain_tp_weed
    data['gain_fp_weed'] = gain_fp_weed
    return data


def grid_gain(data: pd.DataFrame):
    """
    Perform pre-computations on a "grid" of param combinations (i.e. data scenarios). Particularly for the overall gain
    (i.e. gain) in each scenario.
    :param data:
    :return:
    """
    gains = []
    for gain_tp_weed in np.arange(0, 1.1, 0.1):
        for gain_fp_weed in np.arange(-1.1, 0, 0.1):
            df = compute_gain(
                data=deepcopy(data),  # failure to deepcopy -> data changing, end up with list of duplicate dfs
                gain_tp_weed=gain_tp_weed,
                gain_fp_weed=gain_fp_weed
            )
            gains.append(df)

    gains_all = pd.concat(gains)
    gains_all['gain_tp_weed'] = [round(x, 1) for x in gains_all['gain_tp_weed']]
    gains_all['gain_fp_weed'] = [round(x, 1) for x in gains_all['gain_fp_weed']]

    gains_best = gains_all.groupby(['iou_threshold', 'conf_threshold'])['gain_score'].max().reset_index()

    return gains_all, gains_best


def apply_ceiling(df_results: pd.DataFrame, col_to_cap: str = 'fp_weed', col_to_anchor: str = 'tp_weed'):
    """
    Implement ceiling on the feature specified by 'col_to_cap'. Uses max of 'col_to_anchor', so that feature can never
    exceed corresponding value in 'col_to_anchor'
    :param df_results: table of results
    :param col_to_cap: feature to cap
    :param col_to_anchor: feature to base the above cap on
    :return:
    """
    mask = df_results[col_to_cap] < df_results[col_to_anchor]  # mask for rows where col_to_cap < col_to_anchor
    df_results[col_to_cap + '_capped'] = np.where(mask, df_results[col_to_cap], df_results[col_to_anchor])
    return df_results


def extract_process_data(file_dir: str = 'results_iou_conf'):
    """
    Extract TP/FP, AP/mAP data from runs.
    :param file_dir: directory to get raw data from
    :return:
    """
    # get raw data
    files = [f for f in os.listdir(file_dir) if f.endswith('.txt')]  # files with all the data
    df_extracted = extract_data(files=files)  # extract data from files
    df_results = convert_to_numeric(data=df_extracted)  # preprocess extracted data
    df_results = apply_ceiling(
        df_results=df_results,
        col_to_cap='fp_weed',
        col_to_anchor='tp_weed'
    )

    # get gain data
    gains_all, gains_best = grid_gain(data=df_results)

    return df_results, gains_all, gains_best
