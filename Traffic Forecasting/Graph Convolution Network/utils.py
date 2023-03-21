# -*- coding:utf-8 -*-

import os
import numpy as np
import mxnet as mx
import csv


def construct_model(config):
    from models.stsgcn import stsgcn

    module_type = config['module_type']
    act_type = config['act_type']
    temporal_emb = config['temporal_emb']
    spatial_emb = config['spatial_emb']
    use_mask = config['use_mask']
    batch_size = config['batch_size']

    num_of_vertices = config['num_of_vertices']
    num_of_features = config['num_of_features']
    points_per_hour = config['points_per_hour']
    num_for_predict = config['num_for_predict']
    adj_filename = config['adj_filename']
    id_filename = config['id_filename']
    time_step_filename = config['time_step_filename']
    if id_filename is not None:
        if not os.path.exists(id_filename):
            id_filename = None

    points_per_hour = points_per_hour / 3
    num_for_predict = num_for_predict / 3

    adj = get_adjacency_matrix(adj_filename, num_of_vertices,
                               id_filename=id_filename)
    adj_mx = construct_adj(adj, time_step_filename, 4)
    print("The shape of localized adjacency matrix: {}".format(
        adj_mx.shape), flush=True)

    data = mx.sym.var("data")
    label = mx.sym.var("label")
    #adj = mx.sym.Variable('adj', shape=adj_mx.shape,
                          #init=mx.init.Constant(value=adj_mx.tolist()))
    adj = mx.sym.Variable('adj', shape=adj_mx.shape,
                          init=mx.init.Constant(value=adj_mx))
    adj = mx.sym.BlockGrad(adj)
    mask_init_value = mx.init.Constant(value=(adj_mx != 0)
                                       .astype('float32'))

    filters = config['filters']
    first_layer_embedding_size = config['first_layer_embedding_size']

    w_key = mx.sym.var("w_key", shape=(num_of_features, num_of_features))
    w_query = mx.sym.var("w_query", shape=(num_of_features, num_of_features))
    w_val = mx.sym.var("w_val", shape=(num_of_features, num_of_features))

    new_data = mx.sym.reshape(data, (-1, 4 * points_per_hour * num_of_vertices, 1))
    print(new_data.infer_shape(data=(64, 16, 150, 1)))

    keys = mx.sym.dot(new_data, w_key)
    query = mx.sym.dot(new_data, w_query)
    val = mx.sym.dot(new_data, w_val)

    attn_scores = mx.sym.batch_dot(query, mx.sym.transpose(keys, axes=(0, 2, 1)))
    print(attn_scores.infer_shape(data=(64, 16, 150, 1))[1])

    attn_scores_softmax = mx.symbol.softmax(attn_scores, axis=2)

    a = mx.sym.expand_dims(val, axis=2)
    b = mx.sym.expand_dims(attn_scores_softmax, axis=3)
    print(a.infer_shape(data=(64, 16, 150, 1))[1])
    print(b.infer_shape(data=(64, 16, 150, 1))[1])

    weighted_values = mx.sym.broadcast_mul(lhs=a, rhs=b)
    print(weighted_values.infer_shape(data=(64, 16, 150, 1)))
    outputs = mx.symbol.sum(weighted_values, axis=1)
    print(outputs.infer_shape(data=(64, 16, 150, 1)))
    data = mx.sym.reshape(data, (-1, 4 * points_per_hour, num_of_vertices, 1))
    

    if first_layer_embedding_size:
        data = mx.sym.Activation(
            mx.sym.FullyConnected(
                data,
                flatten=False,
                num_hidden=first_layer_embedding_size
            ),
            act_type='relu'
        )
    else:
        first_layer_embedding_size = num_of_features
    net = stsgcn(
        data, adj, label,
        points_per_hour, num_of_vertices, first_layer_embedding_size,
        filters, module_type, act_type,
        use_mask, mask_init_value, temporal_emb, spatial_emb,
        prefix="", rho=1, predict_length=num_for_predict
    )

    assert net.infer_shape(
        data=(batch_size, points_per_hour * 4, num_of_vertices, 1),
        label=(batch_size, num_for_predict, num_of_vertices)
    )[1][1] == (batch_size, num_for_predict, num_of_vertices)
    return net


def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None, time_distance_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


def construct_adj(A, t_s_path, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    '''
    three_list = []
    with open(t_s_path) as f:
        csvread = csv.reader(f)
        three_list = [(row[0], row[1], row[2]) for row in csvread]
    #print('h={},{},{}\n'.format(hd, dd, wd))
    N = len(A)
    adj = np.zeros([N * steps] * 2)

    for i in range(steps):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    # for i in range(N):
    #     for k in range(steps - 1):
    #         adj[k * N + i, (k + 1) * N + i] = 1
    #         adj[(k + 1) * N + i, k * N + i] = 1

    for idx in range(3):
        for i in range(N):
            for k in range(idx + 1):
                adj[k * N + i, (idx + 1) * N + i]
                adj[(idx + 1) * N + i, k * N + i] = three_list[i][idx]

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


def generate_from_train_val_test(data, transformer):
    mean = None
    std = None
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y


def generate_from_data(data, length, point_per_hour, predict_num, transformer):
    mean = None
    std = None
    half = length // 2
    train_line, val_line = int(half + half * 0.6), int(half + half * 0.8)
    for line1, line2 in ((half, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        x, y = generate_seq(data['data'], line1, line2, point_per_hour, point_per_hour, predict_num)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y


def generate_data(graph_signal_matrix_filename, point_per_hour, predict_num, transformer=None):
    '''
    shape is (num_of_samples, 12, num_of_vertices, 1)
    '''
    data = np.load(graph_signal_matrix_filename)
    keys = data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        for i in generate_from_train_val_test(data, transformer):
            yield i
    elif 'data' in keys:
        length = data['data'].shape[0]
        for i in generate_from_data(data, length, point_per_hour, predict_num, transformer):
            yield i
    else:
        raise KeyError("neither data nor train, val, test is in the data")


def generate_seq(data, line1, line2, point_per_hour, train_length, pred_length):
    point_per_day = point_per_hour * 24
    point_per_week = point_per_day * 7

    main_tmp_list = list()
    for i in range(line1, line2 - train_length - pred_length + 1):
        tmp_list = list()
        for j in range(i, i + train_length):
            tmp_list.extend([np.expand_dims(data[j - k], 0) for k in [0, point_per_hour, point_per_day, point_per_week]])
        for j in range(i + train_length, i + train_length + pred_length):
            tmp_list.append(np.expand_dims(data[j], 0))
        tmp_list = np.concatenate(tmp_list, axis=0)
        main_tmp_list.append(np.expand_dims(tmp_list, 0))

    seq = np.concatenate(main_tmp_list, axis=0)[:, :, :, 0: 1]

    return np.split(seq, [4 * train_length], axis=1)

# def generate_seq(data, line1, line2, point_per_hour, train_length, pred_length):
#     seq = np.concatenate([np.expand_dims(
#         data[i: i + train_length + pred_length], 0)
#         for i in range(data.shape[0] - train_length - pred_length + 1)],
#         axis=0)[:, :, :, 0: 1]
#     print(seq.shape)
#     return np.split(seq, [train_length, ], axis=1)


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))
