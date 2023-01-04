#!/bin/bash

mkdir HT\ Data\ Preprocessing/build
cd HT\ Data\ Preprocessing/build
cmake ..
make -j4
cd ../..

mkdir Route\ Planning/build
cd Route\ Planning/build
cmake ..
make -j4
