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
./DFNav_routing --graph_path /path/to/graph --output ./routing_paths.txt o1, d1 [, o2, d2, ...]
```

## Mandatory
`--graph_path`, graph data with three metrics

`--origin`, origin point

`--dest`, destination point

`--output`, output of routing paths

`OD...`, list of origin and destination points
