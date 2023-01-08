import argparse
import os
import importlib

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=str, required=True, help='trajectory data')
    #parser.add_argument("--match", type=str, required=True, help="trajectory matching")
    parser.add_argument("--osm", type=str, required=True, help="openstreetmap data")
    
    args = parser.parse_args()
    os.makedirs('./train', exist_ok=True)
    import my_utils
    my_utils.preprocess(args.osm, args.traj, './train')

    trains = ['passtime', 'flow', 'acc']

    from sys import path as pylib
    import os
    pylib += [os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))), './Traffic Forecasting/Graph Convolution Network')]

    gcn = importlib.import_module('Traffic Forecasting.Graph Convolution Network.train')

    for t in trains:
        gcn.run(['--config', f'./config/{t}.json', '--save', '--save_type', t])
