# Requirements
Python >= 3.7

mxnet-cu100

pytest

graphviz

# Usage
## Quick Start
### Train
```
python3 train.py --config /path/to/config --save
```

### Predict
```
python3 predict.py --config /path/to/config --output predict
```

## Mandatory
`--config`, configure file for running model

`--output`, result name for prediction
