import random
import _pickle as cPickle
import numpy as np
import math
import time
from datetime import datetime
import os
import argparse
import tensorflow as tf
import pandas
from sklearn.cluster import *
from sklearn import preprocessing
import statistics

random.seed(2016)
sampleNum = 10
max_behaviour = 0

def true_data(args):
    dir_path = args.traj
    limit = args.limit
    simData = []
    display_cnt = 0
    for fname in sorted(os.listdir(dir_path), key=lambda f: int(os.path.splitext(f)[0])):
        oneData = []
        with open(os.path.join(dir_path, fname)) as f:
            for line in f:
                sp = line.split(',')
                timestr = sp[1]
                lon = float(sp[2])
                lat = float(sp[3])
                t = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
                ts = int(t.timestamp())
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


def completeTrajectories():
    simTrjss = cPickle.load(open('./true_data/true_trajectories', 'rb'))
    simTrjComps = []
    for simTrjs in simTrjss:
        trjsCom = []
        for i in range(0,len(simTrjs)):
            rec = []
            if i==0:
                # time, locationC, speedC, rotC
                rec = [0,0,0,0]
            else:
                locC = math.sqrt((simTrjs[i][1]-simTrjs[i-1][1])**2+(simTrjs[i][2]-simTrjs[i-1][2])**2)
                rec.append(simTrjs[i][0])
                rec.append(locC)
                rec.append(locC/(simTrjs[i][0]-simTrjs[i-1][0]))
                rec.append(math.atan2((simTrjs[i][2]-simTrjs[i-1][2]), (simTrjs[i][1]-simTrjs[i-1][1])))
            trjsCom.append(rec)
        simTrjComps.append(trjsCom)
    cPickle.dump(simTrjComps,open('./true_data/true_trajectories_complete','wb'))
    return simTrjComps

def computeFeas():
    simTrjCompss = cPickle.load(open('./true_data/true_trajectories_complete', 'rb'))
    simTrjFeas = []
    for simTrjComps in simTrjCompss:
        trjsComfea = []
        for i in range(0,len(simTrjComps)):
            rec = []
            if i==0:
                # time, locationC, speedC, rotC
                rec = [0,0,0,0]
            else:
                locC = simTrjComps[i][1]
                locCrate = locC/(simTrjComps[i][0]-simTrjComps[i-1][0])
                rec.append(simTrjComps[i][0])
                rec.append(locCrate)
                if locCrate<3:
                    rec.append(0)
                    rec.append(0)
                else:
                    rec.append(simTrjComps[i][2]-simTrjComps[i-1][2])
                    rec.append(simTrjComps[i][3]-simTrjComps[i-1][3])
            trjsComfea.append(rec)
        simTrjFeas.append(trjsComfea)
    cPickle.dump(simTrjFeas, open('./true_data/true_trajectories_feas', 'wb'))
    return simTrjFeas

def rolling_window(sample, windowsize = 12000, offset = 6000):
    timeStart = sample[1][0]
    timeLength = sample[len(sample)-1][0]
    windowLength = int ((timeLength - timeStart)/offset)+1
    windows = []
    for i in range(0,windowLength):
        windows.append([])

    for record in sample:
        time = record[0]
        for i in range(0,windowLength):
            if (time>(i*offset+timeStart)) & (time<(i*offset+windowsize+timeStart)):
                windows[i].append(record)
    return windows
    # pass

def behavior_ext(windows):
    behavior_sequence = []
    for window in windows:
        behaviorFeature = []
        records = np.array(window)
        if len(records) != 0:
            # print np.shape(records)
            pd = pandas.DataFrame(records)
            pdd =  pd.describe()
            # print pdd[1][0]
            # for ii in range(1,4):
            #     for jj in range(1,8):
            #         behaviorFeature.append(pdd[ii][jj])
            # behaviorFeature.append(pdd[0][1])
            behaviorFeature.append(pdd[1][1])
            behaviorFeature.append(pdd[2][1])
            behaviorFeature.append(pdd[3][1])
            # behaviorFeature.append(pdd[0][2])
            # behaviorFeature.append(pdd[1][2])
            # behaviorFeature.append(pdd[2][2])
            # behaviorFeature.append(pdd[3][2])
            # behaviorFeature.append(pdd[0][3])
            behaviorFeature.append(pdd[1][3])
            behaviorFeature.append(pdd[2][3])
            behaviorFeature.append(pdd[3][3])
            # behaviorFeature.append(pdd[0][4])
            behaviorFeature.append(pdd[1][4])
            behaviorFeature.append(pdd[2][4])
            behaviorFeature.append(pdd[3][4])
            # behaviorFeature.append(pdd[0][5])
            behaviorFeature.append(pdd[1][5])
            behaviorFeature.append(pdd[2][5])
            behaviorFeature.append(pdd[3][5])
            # behaviorFeature.append(pdd[0][6])
            behaviorFeature.append(pdd[1][6])
            behaviorFeature.append(pdd[2][6])
            behaviorFeature.append(pdd[3][6])
            # behaviorFeature.append(pdd[0][7])
            behaviorFeature.append(pdd[1][7])
            behaviorFeature.append(pdd[2][7])
            behaviorFeature.append(pdd[3][7])

            behavior_sequence.append(behaviorFeature)
    return behavior_sequence

def generate_behavior_sequences():
    f = open('./true_data/true_trajectories_feas', 'rb')
    sim_data = cPickle.load(f)
    behavior_sequences = []

    for sample in sim_data:
        windows = rolling_window(sample)
        behavior_sequence = behavior_ext(windows)
        print(len(behavior_sequence))
        behavior_sequences.append(behavior_sequence)
    fout = open('./true_data/true_behavior_sequences','wb')
    cPickle.dump(behavior_sequences,fout)

def generate_normal_behavior_sequence():
    global max_behaviour
    f = open('./true_data/true_behavior_sequences', 'rb')
    behavior_sequences = cPickle.load(f)

    print(np.shape(behavior_sequences))
    behavior_sequences_normal = []
    templist = []
    for item in behavior_sequences:
        for ii in item:
            templist.append(ii)
        print(len(item))
    print(len(templist))
    min_max_scaler = preprocessing.MinMaxScaler()
    # print np.shape(behavior_sequence)
    templist_normal = min_max_scaler.fit_transform(templist).tolist()
    index = 0
    for item in behavior_sequences:
        behavior_sequence_normal = []
        for ii in item:
            behavior_sequence_normal.append(templist_normal[index])
            index = index + 1
        print(len(behavior_sequence_normal))
        max_behaviour = max(max_behaviour, len(behavior_sequence_normal))
        behavior_sequences_normal.append(behavior_sequence_normal)
    print(index)
    print(np.shape(behavior_sequences_normal))
    print('max={}'.format(max_behaviour))
    fout = open('./true_data/true_normal_behavior_sequences', 'wb')
    cPickle.dump(behavior_sequences_normal, fout)

def trajectory2Vec():
    def loopf(prev, i):
        return prev

    # Parameters
    learning_rate = 0.0001
    training_epochs = 300
    display_step = 100

    # Network Parameters
    # the size of the hidden state for the lstm (notice the lstm uses 2x of this amount so actually lstm will have state of size 2)
    size = 100
    # 2 different sequences total
    batch_size = 1
    # the maximum steps for both sequences is 5
    max_n_steps = 89
    # each element/frame of the sequence has dimension of 3
    frame_dim = 18

    input_length = tf.placeholder(tf.int32)

    initializer = tf.random_uniform_initializer(-1, 1)

    # the sequences, has n steps of maximum size
    # seq_input = tf.placeholder(tf.float32, [batch_size, max_n_steps, frame_dim])
    seq_input = tf.placeholder(tf.float32, [max_n_steps, batch_size, frame_dim])
    # what timesteps we want to stop at, notice it's different for each batch hence dimension of [batch]

    # inputs for rnn needs to be a list, each item/frame being a timestep.
    # we need to split our input into each timestep, and reshape it because split keeps dims by default

    useful_input = seq_input[0:input_length[0]]
    loss_inputs = [tf.reshape(useful_input, [-1])]
    encoder_inputs = [item for item in tf.unstack(seq_input)]
    # if encoder input is "X, Y, Z", then decoder input is "0, X, Y, Z". Therefore, the decoder size
    # and target size equal encoder size plus 1. For simplicity, here I droped the last one.
    decoder_inputs = ([tf.zeros_like(encoder_inputs[0], name="GO")] + encoder_inputs[:-1])
    targets = encoder_inputs

    # basic LSTM seq2seq model
    cell = tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True, use_peepholes=True)
    _, enc_state = tf.contrib.rnn.static_rnn(cell, encoder_inputs, sequence_length=input_length[0], dtype=tf.float32)
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, frame_dim)
    dec_outputs, dec_state = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs, enc_state, cell, loop_function=loopf)


    # flatten the prediction and target to compute squared error loss
    y_true = [tf.reshape(encoder_input, [-1]) for encoder_input in encoder_inputs]
    y_pred = [tf.reshape(dec_output, [-1]) for dec_output in dec_outputs]

    # Define loss and optimizer, minimize the squared error
    loss = 0
    for i in range(len(loss_inputs)):
        loss += tf.reduce_sum(tf.square(tf.subtract(y_pred[i], y_true[len(loss_inputs) - i - 1])))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        input_datas = cPickle.load(open('./true_data/true_normal_behavior_sequences', 'rb'))
        trajectoryVecs = []
        j = 0
        for input_data in input_datas:
            print('Sample:')
            print(j)
            input_len = len(input_data)
            print(input_len)
            defalt = []
            for i in range(0, frame_dim):
                defalt.append(0)
            while len(input_data) < max_n_steps:
                input_data.append(defalt)
            x = np.array(input_data)
            print(np.shape(x[0]))
            x = x.reshape((max_n_steps, batch_size, frame_dim))
            embedding = None
            for epoch in range(training_epochs):
                feed = {seq_input: x, input_length: np.array([input_len])}
                # Fit training using batch data
                _, cost_value, embedding, en_int, de_outs, loss_in = sess.run(
                    [optimizer, loss, enc_state, encoder_inputs, dec_outputs, loss_inputs], feed_dict=feed)
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("logits")
                    a = sess.run(y_pred, feed_dict=feed)
                    print("labels")
                    b = sess.run(y_true, feed_dict=feed)

                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_value))
            trajectoryVecs.append(embedding)
            print("Optimization Finished!")
            j = j + 1
        fout = open('./true_data/true_traj_vec_normal_reverse', 'wb')
        cPickle.dump(trajectoryVecs, fout)


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

def vecClusterAnalysis(args):
    out_folder = args.output
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

    #osmid_dir = r"./T-drive Taxi Trajectories/release/output_osmid"     # Need other pretretment
    osmid_dir = args.traj_mapping
    osmid_dict = dict()

    id_list = []
    display_cnt = 0
    limit = args.limit
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

        with open(os.path.join(out_folder, f'adj_{idx}.txt'), 'w') as f:
            for k, v in result.items():
                f.write(f'{k},{v / avg_of_result * 0.1}\n')

    print(sum)
    add_all = sum[0] + sum[1] + sum[2]
    avg = add_all / 3
    print(avg)
    hd, dd, wd = [1 + (i / add_all) for i in sum]
    print(hd)
    print(dd)
    print(wd)

    with open(os.path.join(out_folder, 'adj.txt'), 'w') as f:
        f.write('{},{},{}\n'.format(hd, dd, wd))


def check_args(args):
    if not os.path.isdir('./true_data'):
        os.makedirs('./true_data')
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    return True


def main(args):
    parser = argparse.ArgumentParser(description='Trajectory clustering for DFNav')
    parser.add_argument('--traj', required=True, metavar='FOLDER',
                        help='Trajectories folder')
    parser.add_argument('--limit', type=int, required=False, metavar='SIZE', default=300,
                        help='Top K')
    parser.add_argument('--traj_mapping', required=True, metavar='FOLDER',
                        help='Osmid directory')
    parser.add_argument('--output', required=True, metavar='FOLDER',
                        help='Output value folder')

    args = parser.parse_args(args)

    if not check_args(args):
        return

    true_data(args)
    completeTrajectories()
    computeFeas()
    generate_behavior_sequences()
    generate_normal_behavior_sequence()
    trajectory2Vec()
    vecClusterAnalysis(args)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
