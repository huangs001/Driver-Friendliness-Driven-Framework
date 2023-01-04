import os.path
import random
import _pickle as cPickle

from sklearn.cluster import *
import statistics
from sklearn import preprocessing
import tensorflow as tf


def split_time_series(ts):
    tss = [ts[0][0]]
    start_idx = [0]
    how_much = []

    hour = 60 * 60
    last = 0

    for i in range(1, len(ts)):
        if ts[i][0] - ts[i - 1][0] > 1 * hour:
            tss.append(ts[i][0])
            how_much.append(i - last)
            start_idx.append(i)
            last = i
    how_much.append(len(ts) - last)
    return zip(tss, start_idx, how_much)


def vecClusterAnalysis():
    trVecs = []
    trs = cPickle.load(open('./true_data/true_traj_vec_normal_reverse', 'rb'))
    inte = []
    for tr in trs:
        trVecs.append(tr[0][0])
    clsnum = 10
    print(1)
    km = KMeans(n_clusters=clsnum, random_state=2016)
    clusters = km.fit(trVecs).labels_.tolist()
    simTrjss = cPickle.load(open('./true_data/true_trajectories', 'rb'))
    time_list = []
    for tri in simTrjss:
        time_list.append(split_time_series(tri))
    cls = [[] for _ in range(clsnum)]
    print(2)
    for idx, ci in enumerate(clusters):
        cls[ci].extend([(t[0], (idx, t[1], t[2])) for t in time_list[idx]])

    hour = 60 * 60
    day = hour * 24
    week = day * 7
    delta = [0.2, 0.2, 0.2]

    sum = [0, 0, 0]
    print(3)

    osmid_dir = r""
    osmid_dict = dict()

    id_list = []
    display_cnt = 0
    limit = 300
    for fname in sorted(os.listdir(osmid_dir), key=lambda f: int(os.path.splitext(f)[0])):
        oneData = 0
        with open(os.path.join(osmid_dir, fname)) as f:
            for _ in f:
                oneData += 1
        if oneData < 5:
            continue
        id_list.append(fname)
        display_cnt += 1
        if display_cnt >= limit:
            break

    print(id_list)

    for idx in id_list:
        with open(os.path.join(osmid_dir, idx)) as f:
            osmid_dict[idx] = [int(x.split(',')[0]) for x in f.read().strip().split('\n')]
    print(osmid_dict.keys())

    result_dict = {}

    for one_cls in cls:
        sort_time = sorted(one_cls)
        for idx, time_step in enumerate([hour, day, week]):
            result_dict.setdefault(idx, {})
            result = result_dict[idx]
            idx2 = 0
            while idx2 < len(sort_time):
                left_all_set = set()
                right_all_set = set()
                i = idx2 + 1
                a, b = 1, 0
                infor = sort_time[idx][1]
                left_all_set.update(osmid_dict[id_list[infor[0]]][infor[1]:infor[1] + infor[2]])

                while i < len(sort_time):
                    if sort_time[i][0] - sort_time[idx2][0] <= 2 * time_step * delta[idx]:
                        a += 1
                        infor = sort_time[i][1]
                        left_all_set.update(osmid_dict[id_list[infor[0]]][infor[1]:infor[1] + infor[2]])
                    if 0 <= sort_time[i][0] - (sort_time[idx2][0] + time_step) <= 2 * time_step * delta[idx]:
                        b += 1
                        infor = sort_time[i][1]
                        right_all_set.update(osmid_dict[id_list[infor[0]]][infor[1]:infor[1] + infor[2]])
                    if sort_time[i][0] - (sort_time[idx2][0] + time_step) > 2 * time_step * delta[idx]:
                        break
                    i += 1
                sum[idx] += min(a, b)
                idx2 = idx2 + 1

                for i in left_all_set:
                    if i in right_all_set:
                        result.setdefault(i, 0)
                        result[i] += 1
    #print(result)
    for idx, _ in enumerate([hour, day, week]):
        result = result_dict[idx]
        if len(result) > 0:
            avg_of_result = statistics.mean(result.values())
        else:
            avg_of_result = 0
        print(avg_of_result)

        with open(f'./true_data/adj_{idx}.txt', 'w') as f:
            for k, v in result.items():
                f.write(f'{k},{v / avg_of_result * 0.1}\n')

    print(sum)
    add_all = sum[0] + sum[1] + sum[2]
    avg = add_all / 3
    print(avg)
    # hd, dd, wd = [1 + ((i - 1.7 * avg) / avg * 0.35) for i in sum]
    hd, dd, wd = [1 + (i / add_all) for i in sum]
    print(hd)
    print(dd)
    print(wd)

    with open('./true_data/adj.txt', 'w') as f:
        f.write('{},{},{}\n'.format(hd, dd, wd))


if __name__ == '__main__':
    vecClusterAnalysis()