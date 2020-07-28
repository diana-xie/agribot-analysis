""" Generate plots of results """
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib
matplotlib.rcParams['font.family'] = 'Franklin Gothic Book'
style.use('ggplot')


def ap_plot(df_results: pd.DataFrame):
    """
    Generates IoU threshold vs. AP/mAP plot
    Parameters
    """
    try:
        # plot - ap
        fig = plt.figure()
        plt.title("Average Precision in Test Set", fontsize=15, weight='bold')
        plt.scatter(df_results['iou_threshold'], df_results['ap_weed'], c="tomato", label='Weed, AP',
                    edgecolors="#2D2926")
        plt.scatter(df_results['iou_threshold'], df_results['ap_corn'], c="dodgerblue", label='Corn, AP',
                    edgecolors="#2D2926")
        plt.scatter(df_results['iou_threshold'], df_results['m_ap'], c="mediumseagreen", label='mAP, across both classes',
                    edgecolors="#2D2926")
        plt.xlabel("IoU Threshold")
        plt.ylabel("Average Precision (AP)")
        plt.legend(loc="upper right")
        fig.savefig('all_MAP.png')  # all means generating pred for range of IoU thresholds + confidence thresholds
    except Exception as ex:
        print("Error in generating 'IoU threshold vs. AP/mAP' plot: {}".format(ex))


def tp_fp_plot(df_results: pd.DataFrame):

    """
    Generate plot of IoU threshold vs. TP weed/corn
    Parameters
    """
    try:
        # plot - fp weeds
        fig = plt.figure()
        plt.title("False Positives (Weeds) in Test Set")
        plt.scatter(df_results['iou_threshold'], df_results['fp_weed'], c="darkred", label='Weed (FP)',
                    edgecolors="#2D2926")
        plt.scatter(df_results['iou_threshold'], df_results['fp_corn'], c="indianred", label='Corn (FP)',
                    edgecolors="#2D2926")
        # plot - tp weeds (on same fig)
        plt.scatter(df_results['iou_threshold'], df_results['tp_weed'], c="darkolivegreen", label='Weed (TP)',
                    edgecolors="#2D2926")
        plt.scatter(df_results['iou_threshold'], df_results['tp_corn'], c="darkseagreen", label='Corn (TP)',
                    edgecolors="#2D2926")
        # labels
        plt.xlabel("IoU Threshold")
        plt.ylabel("# of False Positives (FP)")
        plt.legend(loc="upper left")
        fig.savefig('all_TPFP.png')
    except Exception as ex:
        print("Error in generating 'IoU threshold vs. TP weed/corn' plot: {}".format(ex))


def pr_plot(df_results: pd.DataFrame):
    """
    Generates Precision-Recall plot
    Parameters
    """
    try:
        # PR curve
        fig = plt.figure()
        plt.title("Precision-Recall Curve across different IoU thresholds")
        plt.plot(df_results['recall'], df_results['precision'])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        fig.savefig('Precision-Recall.png')
    except Exception as ex:
        print("Error in generating Precision-Recall plot: {}".format(ex))


def recall_iou_plot(df_results: pd.DataFrame):
    """
    Generate IoU threshold vs. Recall/Precision plot
    """
    try:
        # Recall-IoU curve
        fig = plt.figure()
        plt.title("Recall & Precision vs. IoU Curve")
        plt.plot(df_results['iou_threshold'], df_results['recall'], "-b", label="Recall")
        plt.plot(df_results['iou_threshold'], df_results['precision'], "-r", label="Precision")
        plt.xlabel("IoU threshold")
        plt.ylabel("Recall or Precision (see legend)")
        plt.legend(loc="upper right")
        fig.savefig('Recall-Precision-IoU.png')
    except Exception as ex:
        print("Error in generating 'IoU threshold vs. Recall/Precision' plot: {}".format(ex))


def gain_plot(df_gains: pd.DataFrame):
    """
    Generate  plot
    """
    try:
        # gain curve
        fig = plt.figure()
        plt.title("IoU threshold vs. gain")
        plt.scatter(df_gains['iou_threshold'], df_gains['gain_score'], c="b", label="IoU", edgecolors="#2D2926")
        plt.scatter(df_gains['conf_threshold'], df_gains['gain_score'], c="r", label="Confidence", edgecolors="#2D2926")
        plt.xlabel("IoU/Confidence threshold")
        plt.ylabel("Gain (USD)")
        plt.legend(loc="lower center")
        fig.savefig('IoU_gain.png')
    except Exception as ex:
        print('Error in generating gain plot: {}'.format(ex))


def plot_general(
        df_results: pd.DataFrame,
        df_gains: pd.DataFrame,
        close_plots: bool = False
):
    """
    Generates & saves plots of results
    Parameters
    ----------
    df_results: df of results
    df_gains: df of gains, from df_results
    close_plots: whether or not to close plots immediately after generating
    Returns
    -------
    Generates & saves plots of results
    """
    ap_plot(df_results=df_results)
    tp_fp_plot(df_results=df_results)
    pr_plot(df_results=df_results)
    recall_iou_plot(df_results=df_results)
    gain_plot(df_gains=df_gains)

    if close_plots is True:
        plt.close('all')

