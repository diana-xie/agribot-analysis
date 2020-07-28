import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib
matplotlib.rcParams['font.family'] = 'Franklin Gothic Book'
style.use('ggplot')


def plot_gains_map(
        df_best_gain: pd.DataFrame,
        df_best_map: pd.DataFrame,
        close_plots: bool = False
):
    """
    Generate plots to compare best gain under maximizing gain function vs. map, for each IoU threshold
    Parameters
    ----------
    df_best_gain
    df_best_map
    close_plots: whether or not to close plots immediately after generating
    Returns
    -------

    """
    try:
        # gain curve
        fig = plt.figure()
        plt.title("IoU threshold vs. Gain")
        plt.scatter(df_best_gain['iou_threshold'], df_best_gain['gain_score'], c="b", label="best-gain",
                    edgecolors="#2D2926")
        plt.scatter(df_best_map['iou_threshold'], df_best_map['gain_score'], c="r", label="best-mAP",
                    edgecolors="#2D2926")
        plt.xlabel("IoU threshold")
        plt.ylabel("Gain, Average (USD)")
        plt.legend(loc="lower center")
        fig.savefig('gain_comparisons.png')  # gain_comparisions

        if close_plots is True:
            plt.close('all')

    except Exception as ex:
        print('Error in generating gain plot: {}'.format(ex))


def plot_gains_map_max(gains_all: pd.DataFrame, close_plots: bool = False):
    """
    Generate distribution of gains for each data scenario yielding the best mAP. To illustrate that multiple scenarios
    lead to max mAP, but the distribution of gains from each scenario can vary widely.
    :param gains_all: df of all the gains for each scenario (i.e. Iou threshold x Conf threshold combo)
    :return:
    """

    map_max = gains_all['m_ap'].max()
    gain_max = gains_all['gain_score'].max()

    # dist of gains for the best mAP (max mAP of entire dataset)
    df = gains_all[gains_all['m_ap'] == map_max][['iou_threshold', 'conf_threshold', 'm_ap', 'gain_score']]
    fig = plt.figure()
    sns.distplot(df['gain_score'], color="mediumvioletred")
    plt.title("Distribution of gains for best mAP ({}%)".format(round(map_max * 100, 2)))
    plt.xlabel("Gain (USD)")
    plt.ylabel("Number of scenarios")
    fig.savefig('max_map_gains_distribution.png')  # gain_comparisions

    if close_plots is True:
        plt.close('all')
