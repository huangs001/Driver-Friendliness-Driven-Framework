# Requirements
C++ 17 or higher

# Installation
```
make build
cd build && cmake ..
make
```

# Usage
## Quick Start
```
mkdir output
./HT_process --traj_data /path/to/trajectory --traj_match /path/to/traj_matching --road_data /path/to/road_data --edge_adj /path/to/edge_adj --node_adj /path/to/node_adj --output ./output
```

## Mandatory
`--traj_data`, trajectory data

`--traj_match`, data that trajectory mapping to roads

`--road_data`, road data

`--edge_adj`, edge adjacency matrix of the map

`--node_adj`, node adjacency matrix of the map

`--output`, output folder
