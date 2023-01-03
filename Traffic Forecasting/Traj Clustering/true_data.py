import random
import _pickle as cPickle
import numpy as np
import math
import time
import datetime
import os


def true_data(dir_path):
    simData = []
    display_cnt = 0
    limit = 300
    for fname in sorted(os.listdir(dir_path), key=lambda f: int(os.path.splitext(f)[0])):
        oneData = []
        with open(os.path.join(dir_path, fname)) as f:
            for line in f:
                sp = line.split(',')
                timestr = sp[1]
                lon = float(sp[2])
                lat = float(sp[3])
                t = time.strptime(timestr, '%Y-%m-%d %H:%M:%S')
                ts = int(time.mktime(t))
                if len(oneData) == 0 or oneData[-1][0] != ts:
                    oneData.append([ts, lon, lat])
        if len(oneData) < 5:
            continue
        simData.append(oneData)
        display_cnt += 1
        if display_cnt % 100 == 0:
            print(display_cnt)
        if display_cnt >= limit:
            break

    cPickle.dump(simData,open('./true_data/true_trajectories', 'wb'))


if __name__ == '__main__':
    dir = r'./release/taxi_log_2008_by_id_2'
    true_data(dir)
