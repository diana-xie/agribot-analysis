import os
import re
import pandas as pd
import numpy as np
import cv2
import itertools
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib
matplotlib.rcParams['font.family'] = 'Franklin Gothic Book'
style.use('ggplot')


FILENAME_TEXT = 'results_fps/result_2019_07_24_1_Up_Crash-480x480.txt'
FILENAME_VIDEO = '2019_07_24_1_Up_Crash.MP4'
FILENAME_WEEDS = 'results_fps/weed_frames.csv'
TIME_WITHIN_WEED = 3


# get text data
def extract_data(filename_text: str, filename_weeds: str):
    # open FPS text file
    f = open( filename_text, "r")
    text = f.read()
    # open weed placement file - which was manually generated
    df_weeds = pd.read_csv(filename_weeds)
    df_weeds.dropna(inplace=True)
    df_weeds['frame_num'] = df_weeds['frame_num'].astype(int)
    return text, df_weeds


def get_videoinfo(filename_video: str):
    # get actual number of frames & duration of video
    """https://stackoverflow.com/questions/49048111/how-to-get-the-duration-of-video-using-cv2"""
    cap = cv2.VideoCapture(filename_video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("num frames: {}".format(num_frames))
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print("fps: {}".format(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    minutes = int(duration / 60)
    seconds = duration % 60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
    cap.release()
    return fps, frame_count, duration


# get FPS
def get_fps(text: str):
    df_fps = pd.DataFrame(re.compile('AVG_FPS:(\d+\.\d+)').findall(text)).rename(columns={0: 'avg_fps'})  # avg fps
    df_fps['fps'] = re.compile(r'\nFPS:(\d+\.\d+)').findall(text)  # fps
    df_fps = df_fps.astype(float)  # convert to float
    df_fps = df_fps[df_fps['fps'] != 0]  # remove invalid entries
    df_fps['idx'] = list(range(1, len(df_fps)+1))  # give FPS number
    # df_fps.rename(columns={'index': 'idx'}, inplace=True)
    return df_fps


# get confidence
def get_confidence(text: str):

    text_confidence = text.split('\nFPS')[1:]
    extract_confidence = []

    for frame_num in range(0, len(text_confidence)):
        # weed
        conf_weed = re.compile('weed:.(\d+)').findall(text_confidence[frame_num])
        coord_x_weed = re.compile('weed.+left_x: .(\d+)').findall(text_confidence[frame_num])
        coord_y_weed = re.compile('weed.+top_y: .(\d+)').findall(text_confidence[frame_num])
        size_w_weed = re.compile('weed.+width:  .(\d+)').findall(text_confidence[frame_num])
        size_h_weed = re.compile('weed.+height:  .(\d+)').findall(text_confidence[frame_num])
        extract_weed = zip(conf_weed, ['weed'] * len(conf_weed), [frame_num] * len(conf_weed), coord_x_weed,
                           coord_y_weed, size_w_weed, size_h_weed)
        # corn
        conf_corn = re.compile('corn:.(\d+)').findall(text_confidence[frame_num])
        coord_x_corn = re.compile('corn.+left_x: .(\d+)').findall(text_confidence[frame_num])
        coord_y_corn = re.compile('corn.+top_y: .(\d+)').findall(text_confidence[frame_num])
        size_w_corn = re.compile('corn.+width:  .(\d+)').findall(text_confidence[frame_num])
        size_h_corn = re.compile('corn.+height:  .(\d+)').findall(text_confidence[frame_num])
        extract_corn = zip(conf_corn, ['corn'] * len(conf_corn), [frame_num] * len(conf_corn), coord_x_corn,
                           coord_y_corn, size_w_corn, size_h_corn)
        # save results
        extract_confidence.append(extract_weed)
        extract_confidence.append(extract_corn)

    results = list(itertools.chain(*extract_confidence))
    df = pd.DataFrame(results, columns=['confidence', 'class', 'frame_num', 'coord_x', 'coord_y', 'w', 'h'])
    df['confidence'] = df['confidence'].astype(int)
    df['frame_num'] = df['frame_num'].astype(int)
    df_confidence = df.join(df_weeds[['frame_num', 'arrive_at_weed']].set_index('frame_num'),
                            how='left',
                            on='frame_num')
    return df_confidence


def compute_fpsfeat(df_fps: pd.DataFrame):
    # get FPS, cumulative mean
    df_fps['cummean_raw'] = df_fps['fps'].expanding().mean()  # raw cumulative mean
    warmup_cutoff = df_fps['cummean_raw'].mean() - 2 * df_fps[
        'cummean_raw'].std()  # remove warm-up values, which lowers FPS
    df_fps['cummean_filtered'] = np.nan
    df_fps['cummean_filtered'][df_fps['fps'] > warmup_cutoff] = \
        df_fps['fps'][df_fps['fps'] > warmup_cutoff].expanding().mean()
    return df_fps


def compute_withinweed():
    # calculate the frames in which the robot had time/distance to eliminate weed
    frames_to_weed = round(fps * TIME_WITHIN_WEED)
    frames_to_weed_all = list(zip(df_weeds['frame_num'] - frames_to_weed, df_weeds['frame_num']))
    frames_to_weed_all = [range(x, y+1) for x, y in frames_to_weed_all]
    frames_to_weed_all = [list(x) for x in frames_to_weed_all]
    frames_to_weed_all = list(set(list(itertools.chain(*frames_to_weed_all))))
    return frames_to_weed_all


def plot_fps(df_fps: pd.DataFrame):
    # plot FPS, analysis
    fig = plt.figure()
    # plt.title("FPS in '{}'".format(FILENAME_VIDEO))
    plt.title("FPS in Test Video")
    # plt.plot(df_fps['idx'], df_fps['avg_fps'], "-b", label='FPS (mean)')
    plt.plot(
        df_fps['idx'], df_fps['fps'],
        color='burlywood',
        label='FPS (raw)'
    )
    plt.plot(
        df_fps['idx'], df_fps['cummean_raw'],
        color='darkorange',
        label='FPS, cumulative mean'  # (raw)'
    )
    # plt.plot(
    #     df_fps['idx'], df_fps['cummean_filtered'],
    #     color='darkorange',
    #     label='FPS, cumulative mean (after warmup phase)'
    # )
    plt.xlabel("frame #")
    plt.ylabel("Frames per second (FPS)")
    plt.legend(loc="lower right")
    fig.savefig('results_fps/FPS_analysis_{}.png'.format(FILENAME_VIDEO))


if __name__ == "__main__":

    text, df_weeds = extract_data(FILENAME_TEXT, FILENAME_WEEDS)
    fps, frame_count, duration = get_videoinfo(FILENAME_VIDEO)
    df_fps = get_fps(text)
    df_confidence = get_confidence(text)
    df_fps = compute_fpsfeat(df_fps)
    plot_fps(df_fps=df_fps)
    frames_to_weed_all = compute_withinweed()

    # df = df_confidence[df_confidence['frame_num'].isin(frames_to_weed_all)]
    # df = df[df['class'] == 'weed']

    # df['frame_dist'] = df['frame_num'] - (df['arrive_at_weed'] * df['frame_num'])
    # test = df['frame_dist'].bfill()

    plot_fps()