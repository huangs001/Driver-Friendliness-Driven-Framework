# DFNav
An End-to-End Framework for Enhancing Driver Friendliness in Graph Neural Network Based Navigation Systems

# Requirement
C++ 17 or higher

Python >= 3.7

mxnet-cu100

Tensorflow = 1.15.1

osmnx

pytest

graphviz

pandas

sklearn

# Installation
`bash ./install.sh`

# Preprocess and train model
## Usage
`python preprocess_train.py --traj_data /path/to/trajectory --osm /path/to/osm`

## Mandatory
`--traj_data`, trajectory data

`--osm`, openstreetmap file

# Predict and plan route
## Usage
`python predict_planning.py --od_list /path/to/odlist --output /path/to/output`

## Mandatory
`--od_list`, file contains multiple origin and destination points

`--output`, output of routing paths


