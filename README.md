# DFNav
An End-to-End Framework for Enhancing Driver Friendliness in Graph Neural Network Based Navigation Systems

# Requirement
C++ 17 or higher

Python >= 3.7

mxnet-cu100

Tensorflow = 1.15.1

pytest

graphviz

pandas

sklearn

# Installation
`bash ./install.sh`

# Preprocess and train model
## Usage
`python preprocess_train.py --traj_data /path/to/trajectory --traj_match /path/to/traj_matching --road_data /path/to/road_data --edge_adj /path/to/edge_adj --node_adj /path/to/node_adj`

## Mandatory
`--traj_data`, trajectory data

`--traj_match`, data that trajectory mapping to roads

`--road_data`, road data

`--edge_adj`, edge adjacency matrix of the map

`--node_adj`, node adjacency matrix of the map

# Predict and plan route
## Usage
`python predict_planning.py --od_list /path/to/odlist --osm /path/to/osm --output /path/to/output`

## Mandatory
`--od_list`, file contains multiple origin and destination points

`--osm`, openstreetmap file

`--output`, output of routing paths


