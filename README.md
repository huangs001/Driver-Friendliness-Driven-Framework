# UFN
User friendly navigation framework based on systematic utilization of the user friendliness features from historical traffic data

# 1. Requirements
C++ 17 or higher

Python >= 3.7

mxnet-cu100

Tensorflow = 1.15.1

osmnx

pytest

graphviz

pandas

sklearn

# 2. Installation
`bash ./install.sh`

# 3. Training
`python train.py --traj_data /path/to/trajectory --osm /path/to/osm`

`--traj_data`, trajectory data

`--osm`, OpenStreetMap file

# 4. Preprocessing and traffic forecasting
`python preprocess_forecast.py --traj_data /path/to/trajectory --osm /path/to/osm`

`--traj_data`, trajectory data

`--osm`, OpenStreetMap file

# 5. Route planning
`python route_plan.py --od_list /path/to/odlist --output /path/to/output`

`--od_list`, file containing multiple ODs

`--output`, planned paths

