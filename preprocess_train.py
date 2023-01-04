import argparse
import subprocess
import os
import importlib

import numpy as np
import time
import datetime
import pandas as pd
import csv
import osmnx as ox
import re

def generate_adj(path, node2edge_out, edge_out, edge2edge_out):
    G_BJ = ox.graph_from_xml('../pythonProject/osm/Bj.osm')

    G = ox.graph_from_xml(path)
    # A = nx.adjacency_data(G)

    nodes_adj = dict()
    edge2edge_adj = dict()

    def add_node(node, edge):
        if node not in nodes_adj:
            nodes_adj[node] = set()
        nodes_adj[node].add(edge)

    bad_cnt = 0
    for key, val in G.edges.items():
        u, v, _ = key
        if key not in G_BJ.edges:
            bad_cnt += 1
            continue
        osmid = G_BJ.edges[key]['osmid']
        if isinstance(osmid, list):
            osmid = osmid[0]
        add_node(u, osmid)
        add_node(v, osmid)
    print(bad_cnt)

    for key, val in G.edges.items():
        u, v, _ = key
        if key not in G_BJ.edges:
            continue
        osmid = G_BJ.edges[key]['osmid']
        if isinstance(osmid, list):
            osmid = osmid[0]
        ll = set()

        ll.update([edge for edge in nodes_adj[u] if edge != osmid])
        ll.update([edge for edge in nodes_adj[v] if edge != osmid])

        edge2edge_adj[osmid] = list(ll)

    with open(node2edge_out, 'w') as f:
        for node, adj in nodes_adj.items():
            print(node, file=f)
            print(' '.join({str(idd) for idd in adj}), file=f)

    with open(edge2edge_out, 'w') as f:
        for edge, adj in edge2edge_adj.items():
            print(edge, file=f)
            print(' '.join({str(idd) for idd in adj}), file=f)

    with open(edge_out, 'w') as f:
        for key, val in G.edges.items():
            u, v, _ = key
            if key not in G_BJ.edges:
                continue
            osmid = G_BJ.edges[key]['osmid']
            if isinstance(osmid, list):
                osmid = osmid[0]
            print(osmid, file=f)
            print(' '.join([str(u), str(v)]), file=f)

def output_id(path, main_graph, outpath):
    G1 = ox.graph_from_xml(path)
    G_BJ = main_graph
    # print(len(G1.edges))
    # ox.plot_graph(G1)
    # plt.show()

    with open(outpath, 'w') as f:
        for k, v in G1.edges.items():
            if k not in G_BJ.edges:
                continue
            osmid = G_BJ.edges[k]['osmid']
            print(osmid if not isinstance(osmid, list) else osmid[0], file=f)

def convert(input_ts, input_adj, input_tadj, input_tadj2, input_cnt, output_ts, output_ts_2, output_ts_3, output_df, output_adj, output_adj2, output_adj3, output_id):
    origin_ts = list()
    origin_id = list()
    origin_id_chk = set()
    origin_id_map = dict()
    origin_id_cnt = dict()
    cnt_help = 0
    with open(input_ts) as f:
        for idx, line in enumerate(f):
            if idx % 2 == 0:
                edge_id = int(line)
            else:
                val_list = [float(i) for i in line.split(' ')]
                origin_ts.append(val_list)
                origin_id.append(edge_id)
                origin_id_chk.add(edge_id)
                origin_id_map[edge_id] = cnt_help
                cnt_help += 1
    try:
        with open(input_cnt) as f:
            for line in f:
                sp = line.split()
                origin_id_cnt[int(sp[0])] = int(sp[1])
    except:
        pass

    ts_length = len(origin_ts[0])
    print(ts_length)
    edge_length = len(origin_id)
    print(edge_length)
    data = np.zeros([ts_length, edge_length, 1], dtype=float)

    for idx1 in range(ts_length):
        for idx2 in range(edge_length):
            data[idx1][idx2][0] = origin_ts[idx2][idx1]
    np.savez(output_ts, data=data)
    np.savez(output_ts_2, data=data[ts_length // 2:])

    with open(output_ts_3, 'w') as f:
        for idx1 in range(ts_length // 2, ts_length):
            f.write('{}\n'.format(','.join(str(x[0]) for x in data[idx1])))

    time_fmt = "%Y-%m-%d %H:%M:%S"
    start_time = "2008-02-02 00:00:00"
    start_timestamp = int(datetime.datetime.strptime(start_time, time_fmt).timestamp())
    step_time = 15 * 60
    df_dict = {}

    for i in range(ts_length // 2, ts_length):
        current_time = start_timestamp + i * step_time
        dt = datetime.datetime.fromtimestamp(current_time)
        tmp = []
        for idx, _ in enumerate(origin_id):
            tmp.append(origin_ts[idx][i])
        df_dict[dt] = tmp

    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=origin_id)
    print(df)
    df.to_hdf(output_df, mode='w', key='data')

    origin_adj = list()
    adj_adj = [[0.0 for _ in range(len(origin_id))] for _ in range(len(origin_id))]
    with open(input_adj) as f:
        for idx, line in enumerate(f):
            if idx % 2 == 0:
                edge_id = int(line)
            else:
                if edge_id in origin_id_chk:
                    for succ in [int(i) for i in line.split(' ')]:
                        if succ in origin_id_chk:
                            origin_adj.append((edge_id, succ))
                            row = origin_id_map[edge_id]
                            col = origin_id_map[succ]
                            adj_adj[row][col] = adj_adj[col][row] = 1.0

    def trans(u, v):
        a = origin_id_cnt[u] if u in origin_id_cnt else 0
        b = origin_id_cnt[v] if v in origin_id_cnt else 0
        if a == 0 and b == 0:
            return 1
        return a / (a + b)

    with open(output_adj, 'w') as f:
        f.write('from,to,distance\n')
        for conn in origin_adj:
            f.write('{},{},{}\n'.format(conn[0], conn[1], trans(conn[0], conn[1])))

    with open(output_adj2, 'w') as f:
        for row in adj_adj:
            f.write('{}\n'.format(','.join(str(x) for x in row)))

    with open(output_id, 'w') as f:
        for edge_id in origin_id:
            f.write('{}\n'.format(edge_id))

    hh, dd, ww = 1, 1, 1
    with open(input_tadj) as f:
        hh, dd, ww = [float(x) for x in f.read().strip().split(',')]

    val_dict = {}
    for idx, iadj in enumerate(input_tadj2):
        val_dict.setdefault(idx, {})
        with open(iadj) as f:
            reader = csv.reader(f)
            val_dict[idx] = {int(row[0]): float(row[1]) for row in reader}

    with open(output_adj3, 'w') as f:
        for idd in origin_id:
            tmp_list = []
            for idx, v in enumerate([hh, dd, ww]):
                if idd in val_dict[idx]:
                    tmp_list.append(str(v + val_dict[idx][idd]))
                else:
                    tmp_list.append(str(v))
            f.write(f'{",".join(tmp_list)}\n')


def convert_i(input1, input2, output, i):
    small = "small{}".format(i)
    data_type = ["flow", "passtime", "acc"]

    for t in data_type:
        os.makedirs(r"{}/{}/{}".format(output, small, t), exist_ok=True)
        convert(r"{}/{}/{}/data.txt".format(input1, small, t),
                r"{}/{}/filter/{}.txt".format(input1, small, t),
                r"{}/adj.txt".format(input2),
                [
                    r"{}/adj_0.txt".format(input2),
                    r"{}/adj_1.txt".format(input2),
                    r"{}/adj_2.txt".format(input2)
                ],
                "",
                r"{}/{}/{}/{}.npz".format(output, small, t, t),
                r"{}/{}/{}/{}2.npz".format(output, small, t, t),
                r"{}/{}/{}/{}2.csv".format(output, small, t, t),
                r"{}/{}/{}/{}2.h5".format(output, small, t, t),
                r"{}/{}/{}/{}.csv".format(output, small, t, t),
                r"{}/{}/{}/{}_sparse.csv".format(output, small, t, t),
                r"{}/{}/{}/{}_time.txt".format(output, small, t, t),
                r"{}/{}/{}/{}_id.txt".format(output, small, t, t))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=str, help='trajectory data')
    parser.add_argument("--match", type=str, required=True, help="trajectory matching")
    parser.add_argument("--osm", type=str, required=True, help="openstreetmap data")
    
    args = parser.parse_args()

    road_path = './road.txt'
    match_folder = args.match
    file_list = os.listdir(match_folder)
    pattern = re.compile('<.*>')
    road_length = {}
    os.makedirs('./traj_osmid', exist_ok=True)
    for file in file_list:
        with open(os.path.join(match_folder, file)) as f, open(f'./traj_osmid/{file}', 'w') as f2:
            for l in f:
                l = pattern.sub('\"\"', l)
                data = eval(l)
                osmid = data['osmid']

                if not isinstance(data['osmid'], list):
                    osmid = [osmid]
                f2.write(f'{osmid[0]}\n')
                for oid in osmid:
                    road_length[oid] = (data['length'], data['maxspeed'])

    with open(os.path.join(road_path), 'w') as f:
        for d in road_length.items():
            print('{},{},{}'.format(d[0], d[1][0], d[1][1]), file=f)

    print("Done")

    print("Generating adj...")
    main_graph = args.osm
    generate_adj(main_graph, main_graph,
                    os.path.join('node.txt'),
                    os.path.join('edge.txt'),
                    os.path.join('edge2edge.txt'))
    print("Done")

    os.makedirs('./output/small1/acc', exist_ok=True)
    os.makedirs('./output/small1/flow', exist_ok=True)
    os.makedirs('./output/small1/passtime', exist_ok=True)
    os.makedirs('./output/small1/trjCls', exist_ok=True)
    os.makedirs('./output/small1/filter', exist_ok=True)

    subprocess.run(['./HT Data Preprocessing/build/HT_process', '--traj_data', args.traj, '--traj_match', './traj_osmid',
                    '--road_data', road_path, '--edge_adj', 'edge.txt', '--node_adj', 'node.txt', '--output', './output'])

    cluster = importlib.import_module('Traffic Forecasting.Traj Clustering.cluster')
    cluster.main(['--traj', args.traj, '--traj_mapping', './traj_osmid', '--output', './traj_result'])

    convert_i('./output', './traj_result', './data', 1)

    trains = ['passtime', 'flow', 'acc']

    gcn = importlib.import_module('Traffic Forecasting.Graph Convolution Network.train')

    for t in trains:
        gcn.run(['--config', f'./{t}.json', '--save', '--save_type', t])
