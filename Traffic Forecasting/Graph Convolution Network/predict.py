# -*- coding:utf-8 -*-

import time
import json
import argparse

import numpy as np
import mxnet as mx

from utils import (construct_model, generate_data,
                   masked_mae_np, masked_mape_np, masked_mse_np)

def run(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    parser.add_argument("--test2", action="store_true", help="test program")
    parser.add_argument("--type", type=str, help='', required=True)
    parser.add_argument("--output", type=str, help='output result', required=True)
    args = parser.parse_args(args)

    config_filename = args.config

    with open(config_filename, 'r') as f:
        config = json.loads(f.read())

    num_of_vertices = config['num_of_vertices']
    graph_signal_matrix_filename = config['graph_signal_matrix_filename']

    point_per_hour = config['points_per_hour']

    point_per_day = point_per_hour * 24
    point_per_week = point_per_day * 7

    epochs = int(config['epochs'])
    if args.test2:
        epochs = 2

    mod = mx.mod.Module.load(args.type, epochs)

    mod.bind(
        for_training=False,
        data_shapes=[(
            'data',
            (1, 4 * config['points_per_hour'], num_of_vertices, 1)
        ), ]
    )

    data = np.load(graph_signal_matrix_filename)
    data = data['data']
    main_tmp_list = list()
    tmp_list = list()
    for i in range(-1, -1 - 4, -1):
        tmp_list.extend([np.expand_dims(data[-1 + i - k], 0) for k in [0, point_per_hour, point_per_day, point_per_week]])
    tmp_list = np.concatenate(tmp_list, axis=0)
    main_tmp_list.append(np.expand_dims(tmp_list, 0))
    seq = np.concatenate(main_tmp_list, axis=0)[:, :, :, 0: 1]
    print(np.shape(seq))

    predict = mod.predict(seq)
    print(predict)

    import os
    np.save(os.path.join('./forecast', args.output), predict.asnumpy())

if __name__ == '__main__':
    import sys
    run(sys.argv[1:])
