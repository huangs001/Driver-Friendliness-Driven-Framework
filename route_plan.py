import argparse
import subprocess

import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--od_list", type=str, required=True, help='file to od_list')
    parser.add_argument("--output", type=str, required=True, help="output")
    
    args = parser.parse_args()

    base_dir = './forecast'
    graph_path = os.path.join(base_dir, 'cch1.txt')
    
    cmd = ['./Route Planning/build/DFNav_routing', '--graph_path', graph_path, '--output', args.output]

    with open(args.od_list) as f:
        for line in f:
            sp = line.split()
            cmd.append(str(int(sp[0])))
            cmd.append(str(int(sp[1])))
    print(cmd)

    subprocess.run(cmd)
