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
`python preprocess_train.py --traj_data /path/to/trajectory --traj_match /path/to/traj_matching --osm /path/to/osm`

## Mandatory
`--traj_data`, trajectory data

`--traj_match`, data that trajectory mapping to roads

`--osm`, openstreetmap file

# Predict and plan route
## Usage
`python predict_planning.py --od_list /path/to/odlist --osm /path/to/osm --output /path/to/output`

## Mandatory
`--od_list`, file contains multiple origin and destination points

`--osm`, openstreetmap file

`--output`, output of routing paths


