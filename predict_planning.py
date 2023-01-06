import argparse
import importlib
import subprocess
import numpy as np

import osmnx as ox
import pickle
import statistics

def load_ts(input_ts):
    print(f'Loading {input_ts}')
    origin_ts = list()
    origin_id_map = dict()
    cnt_help = 0
    with open(input_ts) as f:
        for idx, line in enumerate(f):
            if idx % 2 == 0:
                edge_id = int(line)
            else:
                val_list = [float(i) for i in line.split(' ')]
                origin_ts.append(val_list)
                origin_id_map[edge_id] = cnt_help
                cnt_help += 1
    print(f'Loaded {input_ts}')
    return origin_ts, origin_id_map


def construct_adj(path, G_P, input_ts, output_path):
    ts, ts_map = zip(*[load_ts(x) for x in input_ts])

    nodes_adj = dict()

    def add_node(u, v, weight):
        nodes_adj.setdefault(u, list())
        nodes_adj.setdefault(v, list())

        nodes_adj[u].append((v, weight))

    x, y, z = [sum(statistics.mean(ts[i][k]) for k in ts_map[i].values()) for i in range(len(input_ts))]
    print(x, y, z)
    weights = [1, 1, y / z]

    G = ox.graph_from_xml(path)
    # A = nx.adjacency_data(G)

    bad_cnt = 0
    default_cnt = 0

    default_list = [a / b for a, b in zip([x, y, z], [len(ts) for ts in input_ts])]
    default_list = [weights[i] * default_list[i] for i in range(len(default_list))]
    print(default_list)
    small_to_origin = dict()
    for_cch = list()
    road_list = []

    chk = set()

    for key, val in G.edges.items():
        u, v, _ = key
        if key not in G_P.edges:
            bad_cnt += 1
            #continue
            osmid = 9999999
            length = G.edges[key]['length']
        else:
            path2 = G_P.edges[key]
            osmid = path2['osmid']
            length = path2['length']

        if isinstance(osmid, list):
            osmid = osmid[0]

        val_list = []

        for i in range(len(input_ts)):
            if osmid in ts_map[i]:
                ori_val = statistics.mean(ts[i][ts_map[i][osmid]]) * weights[i]
                if ori_val > x:
                    val_list.append(ori_val)
                else:
                    val_list.append(ori_val)
            else:
                val_list.append(default_list[i])
                default_cnt += 1
       
        #add_node(v, u, (osmid, length, *[x for x in val_list]))
        small_id = G.edges[key]['osmid']
        if isinstance(small_id, list):
            small_id = small_id[0]
        if small_id in chk:
            continue
        chk.add(small_id)
        add_node(u, v, (osmid, length, *[x for x in val_list]))
        small_to_origin[small_id] = (osmid, length, *[x for x in val_list])
        for_cch.append((small_id, osmid, u, v, length, val_list[0] + 0.5 * length, val_list[1] * length, val_list[2] * length))
        road_list.append((val_list[0] + 0.5 * length, small_id))
    
    road_list.sort(reverse=False)
    current_similar = 0

    def is_road_similar(a, b):
        return abs(b - a) / a < 0.5

    while current_similar < len(road_list):
        i = current_similar + 1
        while i < len(road_list) and is_road_similar(road_list[current_similar][0], road_list[i][0]):
            road_list[i] = (road_list[current_similar][0], road_list[i][1])
            i += 1
        current_similar = i
    
    blur = {k: int(max(1.0, round(v))) for v, k in road_list}

    print(bad_cnt)
    print(default_cnt)
    nodes_adj_simple = nodes_adj

    with open(f'{output_path}', 'w') as f:
        for l in for_cch:
            f.write(f'{l[0]},{l[1]},{l[2]},{l[3]},{l[4]},{l[5]},{blur[l[0]]},{l[6]},{l[7]}\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--od_list", type=str, required=True, help='file to od_list')
    #parser.add_argument("--osm", type=str, required=True, help='Openstreetmap file')
    parser.add_argument("--output", type=str, required=True, help="output")
    
    args = parser.parse_args()
    ###
    trains = ['passtime', 'flow', 'acc']
    
    from sys import path as pylib
    import os
    pylib += [os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))), './Traffic Forecasting/Graph Convolution Network')]

    gcn = importlib.import_module('Traffic Forecasting.Graph Convolution Network.predict')

    for t in trains:
        gcn.run(['--config', f'./config/{t}.json', '--type', t, '--output', t])
        data = np.load(f'./{t}.npy')
        #data = np.random.randint(0, 10, (1, 4, 150))

        with open(f'./output/small1/{t}/data.txt') as f1, open(f'./output/small1/{t}/data2.txt') as f2, open(f'./{t}001.txt', 'w') as f3:
            id_list = list()
            for idx, line in enumerate(f1):
                if idx % 2 == 0:
                    id_list.append(int(line.strip()))
            print(len(id_list))
            buf_line = ''
            for idx, line in enumerate(f2):
                if idx % 2 == 0:
                    f3.write(f'{line}')
                    buf_line = line
                else:
                    if int(buf_line.strip()) in id_list:
                        jdx = id_list.index(int(buf_line.strip()))
                        f3.write(f'{int(10 * data[0][0][jdx])}\n')
                    else:
                        f3.write(f'{10 * int(line.split()[0])}\n')
    
    osm = './tmp_map.osm'
    G_P = ox.graph_from_xml(osm)
    construct_adj(osm, G_P,
                  [r'./flow001.txt',
                   r'./passtime001.txt',
                   r'./cch1001.txt'], './cch1.txt')

    
    cmd = ['./Route Planning/build/DFNav_routing', '--graph_path', './cch1.txt', '--output', args.output]

    with open(args.od_list) as f:
        for line in f:
            sp = line.split()
            cmd.append(str(int(sp[0])))
            cmd.append(str(int(sp[1])))
    print(cmd)

    subprocess.run(cmd)
